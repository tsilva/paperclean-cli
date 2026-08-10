"""Conservative work and cost projections for paid model calls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from paperclean.errors import GlobalOpenRouterError

# PaperClean sends one full-page comparison plus four regional comparisons.
REVIEW_VIEWS_PER_ATTEMPT = 5

# A generation can be retried once with a smaller reference image. Each review
# can be retried once if the first structured response is invalid.
GENERATION_REQUESTS_PER_RECOVERY_ATTEMPT = 2
REVIEW_REQUESTS_PER_RECOVERY_ATTEMPT = REVIEW_VIEWS_PER_ATTEMPT * 2

# Conservative token assumptions based on the maximum image sizes PaperClean
# sends and high-quality GPT Image output. They deliberately favor overestimation.
GENERATION_INPUT_IMAGE_TOKENS = 16_384
GENERATION_INPUT_TEXT_TOKENS = 512
GENERATION_OUTPUT_IMAGE_TOKENS = 7_034
REVIEW_INPUT_IMAGE_TOKENS = 8_192
REVIEW_INPUT_TEXT_TOKENS = 1_024
REVIEW_MAX_COMPLETION_TOKENS = 4_096


@dataclass(frozen=True, slots=True)
class UnitPrices:
    """USD prices per token for the billable modalities PaperClean uses."""

    input_text: Decimal | None = None
    input_image: Decimal | None = None
    output_text: Decimal | None = None
    output_image: Decimal | None = None


@dataclass(frozen=True, slots=True)
class WorkEstimate:
    generations: int
    reviews: int
    cost_usd: Decimal

    @property
    def paid_calls(self) -> int:
        return self.generations + self.reviews


@dataclass(frozen=True, slots=True)
class CostProjection:
    document_total: int
    page_total: int
    max_attempts: int
    image_model: str
    image_provider: str
    review_enabled: bool
    review_model: str | None
    review_provider: str | None
    one_pass: WorkEstimate
    configured_max: WorkEstimate
    recovery_ceiling: WorkEstimate
    account_remaining_usd: Decimal | None
    key_remaining_usd: Decimal | None
    key_unlimited: bool
    soft_limit_usd: Decimal | None

    @property
    def effective_available_usd(self) -> Decimal | None:
        values = [
            value
            for value in (self.account_remaining_usd, self.key_remaining_usd)
            if value is not None
        ]
        return min(values) if values else None


def _price(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result >= 0 else None


def parse_unit_prices(raw: Any) -> UnitPrices:
    """Normalize both OpenRouter endpoint-pricing response shapes."""

    values: dict[str, Decimal] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or item.get("unit") not in (None, "token"):
                continue
            billable = str(item.get("billable") or "")
            cost = _price(item.get("cost_usd"))
            if billable and cost is not None:
                values[billable] = cost
    elif isinstance(raw, dict):
        for key, value in raw.items():
            cost = _price(value)
            if cost is not None:
                values[str(key)] = cost

    prompt = values.get("prompt")
    completion = values.get("completion")
    return UnitPrices(
        input_text=values.get("input_text", prompt),
        input_image=values.get("input_image", prompt),
        output_text=values.get("output_text", completion),
        output_image=values.get("output_image"),
    )


def _required(value: Decimal | None, description: str) -> Decimal:
    if value is None:
        raise GlobalOpenRouterError(
            f"selected endpoint does not publish {description} pricing required for preflight"
        )
    return value


def build_cost_projection(
    *,
    document_total: int,
    page_total: int,
    max_attempts: int,
    image_model: str,
    image_provider: str,
    image_prices: UnitPrices,
    review_enabled: bool,
    review_model: str | None,
    review_provider: str | None,
    review_prices: UnitPrices | None,
    account_remaining_usd: Decimal | None,
    key_remaining_usd: Decimal | None,
    key_unlimited: bool,
    soft_limit_usd: Decimal | None,
) -> CostProjection:
    generation_cost = (
        _required(image_prices.input_image, "image-input") * GENERATION_INPUT_IMAGE_TOKENS
        + _required(image_prices.input_text, "text-input") * GENERATION_INPUT_TEXT_TOKENS
        + _required(image_prices.output_image, "image-output") * GENERATION_OUTPUT_IMAGE_TOKENS
    )
    review_cost = Decimal("0")
    if review_enabled:
        if review_prices is None or review_model is None or review_provider is None:
            raise GlobalOpenRouterError(
                "review endpoint metadata is required when review is enabled"
            )
        review_image_price = (
            review_prices.input_image
            if review_prices.input_image is not None
            else review_prices.input_text
        )
        review_cost = (
            _required(review_image_price, "review image-input") * REVIEW_INPUT_IMAGE_TOKENS
            + _required(review_prices.input_text, "review text-input") * REVIEW_INPUT_TEXT_TOKENS
            + _required(review_prices.output_text, "review text-output")
            * REVIEW_MAX_COMPLETION_TOKENS
        )

    def estimate(generations: int, reviews: int) -> WorkEstimate:
        return WorkEstimate(
            generations=generations,
            reviews=reviews,
            cost_usd=generation_cost * generations + review_cost * reviews,
        )

    configured_pages = page_total * max_attempts
    review_multiplier = REVIEW_VIEWS_PER_ATTEMPT if review_enabled else 0
    recovery_review_multiplier = REVIEW_REQUESTS_PER_RECOVERY_ATTEMPT if review_enabled else 0
    return CostProjection(
        document_total=document_total,
        page_total=page_total,
        max_attempts=max_attempts,
        image_model=image_model,
        image_provider=image_provider,
        review_enabled=review_enabled,
        review_model=review_model,
        review_provider=review_provider,
        one_pass=estimate(page_total, page_total * review_multiplier),
        configured_max=estimate(
            configured_pages,
            configured_pages * review_multiplier,
        ),
        recovery_ceiling=estimate(
            configured_pages * GENERATION_REQUESTS_PER_RECOVERY_ATTEMPT,
            configured_pages * recovery_review_multiplier,
        ),
        account_remaining_usd=account_remaining_usd,
        key_remaining_usd=key_remaining_usd,
        key_unlimited=key_unlimited,
        soft_limit_usd=soft_limit_usd,
    )
