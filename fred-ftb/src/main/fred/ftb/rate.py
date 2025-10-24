from dataclasses import dataclass
from typing import Optional

from fred.settings import logger_manager
from fred.ftb.compounding.frequency import CompoundingFrequency
from fred.ftb.compounding.unit import CompoundingUnit

logger = logger_manager.get_logger(name=__name__)


@dataclass(frozen=True, slots=True)
class Rate:
    value: float
    comp_freq: CompoundingFrequency
    expr_unit: CompoundingUnit

    @classmethod
    def auto(
            cls,
            value: float,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
    ) -> 'Rate':
        """Automatically creates a Rate with annual compounding frequency and unit."""
        return cls(
            value=value,
            comp_freq=comp_freq or CompoundingFrequency.ANNUAL,
            expr_unit=expr_unit or CompoundingUnit.YEARS,
        )
    
    def clone(
            self, 
            value: Optional[float] = None,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
    ) -> 'Rate':
        """Clones the Rate with optional modifications."""
        return Rate(
            value=value if value is not None else self.value,
            comp_freq=comp_freq if comp_freq is not None else self.comp_freq,
            expr_unit=expr_unit if expr_unit is not None else self.expr_unit,
        )

    def percent(self, round_digits: int = 6) -> float:
        """Returns the rate as a percentage."""
        return 100 * round(self.value, round_digits)

    def get_base(self) -> 'Rate':
        """Returns the base rate per compounding period considering how the rate is expressed and the compounding frequency.
        
        Consider a rate "r" expressed in "e" units (expr_unit) and compounded at frequency "c" (comp_freq).
        The base rate per compounding period is adjusted by the ratio of periods per year between the expression unit and the compounding frequency.
        For example, if a rate is expressed in months (expr_unit = MONTHS) but compounded quarterly (comp_freq = QUARTERLY),
        the adjustment factor would be (12 months/year) / (4 quarters/year) = 3 months per quarter.
        Thus, the base rate per compounding period would be b = r * 3.

        Consider that the new expression unit becomes the compounding frequency's unit after this adjustment.
        """
        adjustment = self.expr_unit.compounding_periods_per_year / self.comp_freq.periods_per_year
        return self.clone(
            value=self.value * adjustment,
            comp_freq=self.comp_freq,
            expr_unit=CompoundingUnit(self.comp_freq)
        )

    def effective_annual(self) -> 'Rate':
        """Returns the effective annual rate.
        The effective annual rate (EAR) is the annual representation of the interest rate that is adjusted for compounding over a year.
        Therefore, it standardizes the rate to an annual basis, allowing for easier comparison between rates with different compounding frequencies.
        """
        # Convert the rate to a decimal form based on its expression unit
        rate = self.get_base()
        # Calculate the effective annual rate
        return self.__class__(
            value=(1 + rate.value) ** self.comp_freq.periods_per_year - 1,
            comp_freq=CompoundingFrequency.ANNUAL,
            expr_unit=CompoundingUnit.YEARS,
        )

    def expr_annual(self) -> 'Rate':
        return self.expr_as(unit=CompoundingUnit.YEARS)

    def expr_as(self, unit: CompoundingUnit) -> 'Rate':
        """Returns the rate expressed in the specified units.
        Converts the rate to be expressed in terms of the specified unit.
        This doesn't change the compounding frequency, only the expression units."""
        if unit == self.expr_unit:
            return self.clone()
        rate = self.get_base()
        adjustment = rate.comp_freq.periods_per_year / unit.compounding_periods_per_year
        return self.clone(
            value=rate.value * adjustment,
            comp_freq=self.comp_freq,  # Compounding frequency remains the same
            expr_unit=unit,  # Update expression unit
        )

    def freq_as(self, frequency: CompoundingFrequency, expr_unit: int = 0) -> 'Rate':
        """Returns the rate with the specified compounding frequency.
        Converts the rate to have the specified compounding frequency while keeping the expression units the same.
        
        The idea is to transform from a rate "r" expressed in "e" units (expr_unit) and compounded at frequency "c" (comp_freq)
        equivalent to "n" periods per year to a new rate "i" expressed in the same "x" units but compounded at a different
        frequency "f" (frequency) equivalent to "m" periods per year. We can achieve this by solving for an equivalent "annual" rate (EAR). 
        
        (1 + base_r)  ^ n - 1 = (1 + base_i) ^ m - 1
        => (1 + base_r)  ^ n = (1 + base_i) ^ m
        => (1 + base_r)  ^ (n/m) = (1 + base_i)
        => base_i = (1 + base_r) ^ (n/m) - 1

        Once we have the new base (i.e., base_i), we can adjust to the desired expression units as needed. The following options are available for convinience:
        - expr_unit = -1: Keep reference expression unit (original object)
        - expr_unit = 0: Use standard expression unit (i.e., annual representation)
        - expr_unit = 1: Match expression unit to compounding frequency

        By default, expr_unit = 0 is used (i.e., annual representation regardless of the compounding frequency).
        """
        if frequency == self.comp_freq:
            return self.clone()
        base = self.get_base()
        # Get the new rate on the desired frequency and the equivalent expression units (i.e., new base)
        rate = self.clone(
            value=(1 + base.value) ** (base.comp_freq.periods_per_year / frequency.periods_per_year ) - 1,
            comp_freq=frequency,
            expr_unit=CompoundingUnit(frequency),
        )
        match expr_unit:
            # Keep reference expression unit (original object)
            case -1:
                return rate.expr_as(unit=self.expr_unit)
            # DEFAULT: Use standard expression unit (i.e., annual representation)
            case 0:  
                return rate.expr_as(unit=CompoundingUnit.YEARS)
            # Match expression unit to compounding frequency
            case 1:
                return rate.expr_as(unit=CompoundingUnit(frequency))
            case _:
                logger.warning(f"Unknown expr_unit value '{expr_unit}'. Returning rate with annual expression unit.")
                return rate.expr_as(unit=CompoundingUnit.YEARS)

    def factor(self, periods: float) -> float:
        """Calculates the growth factor for a given number of periods."""
        base = self.get_base()
        return (1 + base.value) ** periods

    def __call__(self, amount: float, periods: float) -> float:
        """Calculates the growth factor for a given number of periods."""
        return amount * self.factor(periods)
    
    def get_period_multiplier(self, period_units: Optional[CompoundingUnit] = None) -> float:
        """Returns the multiplier to convert periods in the given units to periods in the rate's compounding frequency."""
        if period_units is None:
            return 1.0
        return self.comp_freq.periods_per_year / period_units.compounding_periods_per_year

    def apply(self, amount: float, periods: int, period_units: Optional[CompoundingUnit] = None) -> float:
        """Applies the rate to a given amount over a number of periods."""
        multiplier = self.get_period_multiplier(period_units=period_units)
        return self(amount=amount, periods=periods * multiplier)

    def is_equivalent(self, other: 'Rate', tol: Optional[float] = None) -> bool:
        """Checks if two rates are equivalent in terms of effective annual rate."""
        tol = tol or 1e-5
        return abs(self.effective_annual().value - other.effective_annual().value) <= tol
