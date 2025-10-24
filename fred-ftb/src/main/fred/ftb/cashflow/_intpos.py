import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from fred.settings import logger_manager
from fred.ftb.compounding.unit import CompoundingUnit
from fred.ftb.cashflow.interface import CashFlowInterface, ProjectStructInterface


logger = logger_manager.get_logger(name=__name__)


@dataclass(slots=True, frozen=True)
class PCF(CashFlowInterface[int]):
    amount: float
    t: int  # time period in which the cash flow occurs

    def __post_init__(self):
        if self.t < 0:
            raise ValueError("Time period 't' must be non-negative.")

    @property
    def default_period_unit(self) -> CompoundingUnit:
        return CompoundingUnit.YEARS

    @property
    def description(self) -> str:
        import textwrap
        return textwrap.dedent(
            """
            PCF Cash Flow - Position-based Cash Flow (Integer Time Period)
            + Amount: {amount}
            + Time Period (t): {t}
            """.format(amount=self.amount, t=self.t)
        ).strip()

    @property
    def caperiod_present_ref(self) -> int:
        return 0

    @staticmethod
    def struct_cls() -> type['PCFProjectStruct']:
        return PCFProjectStruct

    @staticmethod
    def struct(flows: Optional[list['PCF']] = None) -> 'PCFProjectStruct':
        return PCFProjectStruct(
            flows={cf.t: cf for cf in flows or []}
        )

    def caperiod_diff(self, end: int) -> int:
        return end - self.t
    
    def caperiod_add(self, periods: int) -> int:
        return self.t + periods
    
    def to_xcf(self, reference_date: dt.date, as_current: bool = False) -> CashFlowInterface[dt.date]:
        from fred.ftb.cashflow._datepos import XCF
        return XCF(
            amount=self.amount,
            t=reference_date if as_current else (
                reference_date + dt.timedelta(days=self.t)
            )
        )


@dataclass(slots=True, frozen=True)
class PCFProjectStruct(ProjectStructInterface[int]):
    flows: dict[int, PCF] = field(default_factory=dict)

    def to_xcf(self, start_date: Optional[dt.date | str] = None) -> ProjectStructInterface[dt.date]:
        from fred.ftb.cashflow._datepos import XCFProjectStruct
        from fred.utils.dateops import datetime_utcnow
        base = start_date or datetime_utcnow().date()
        if isinstance(base, str):
            base = dt.date.fromisoformat(base)
        struct = XCFProjectStruct.empty()
        for cf in self.flows.values():
            struct = struct.add_cashflow(cf.to_xcf(reference_date=base, as_current=False))
        return struct
