import enum


class CompoundingFrequency(enum.Enum):
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365
    DAILY_360 = 360

    def periods_per(self, other: 'CompoundingFrequency') -> float:
        """Returns the number of compounding periods of this frequency per period of the other frequency."""
        return other.value / self.value

    @property
    def periods_per_year(self) -> int:
        """Returns the number of compounding periods per year for this frequency."""
        return self.value

    def __str__(self) -> str:
        return f"FREQ.{self.name}"

    def __repr__(self) -> str:
        return str(self)
