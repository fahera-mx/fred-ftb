import enum

from fred.ftb.compounding.frequency import CompoundingFrequency


class CompoundingUnit(enum.Enum):
    YEARS = CompoundingFrequency.ANNUAL
    HALF_YEARS = CompoundingFrequency.SEMI_ANNUAL
    QUARTERS = CompoundingFrequency.QUARTERLY
    MONTHS = CompoundingFrequency.MONTHLY
    WEEKS = CompoundingFrequency.WEEKLY
    DAYS = CompoundingFrequency.DAILY

    @property
    def compounding_periods_per_year(self) -> int:
        """Returns the number of compounding periods per year for this unit."""
        return self.value.periods_per_year

    def as_unit(self, other: 'CompoundingUnit') -> float:
        """Returns the number of this unit per other unit."""
        return self.value.periods_per(other.value)

    def as_freq(self) -> CompoundingFrequency:
        return self.value

    def __str__(self) -> str:
        return f"UNIT.{self.name}"

    def __repr__(self) -> str:
        return str(self)
