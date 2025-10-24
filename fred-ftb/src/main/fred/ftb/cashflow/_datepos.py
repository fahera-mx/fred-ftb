import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from fred.settings import logger_manager
from fred.ftb.compounding.unit import CompoundingUnit
from fred.ftb.cashflow.interface import CashFlowInterface, ProjectStructInterface


logger = logger_manager.get_logger(name=__name__)


@dataclass(slots=True, frozen=True)
class XCF(CashFlowInterface[dt.date]):
    amount: float
    t: dt.date  # time period in which the cash flow occurs

    @property
    def description(self) -> str:
        import textwrap
        return textwrap.dedent(
            """
            XCF Cash Flow - Date-based Cash Flow (Date Time Period)
            + Amount: {amount}
            + Time Period (t): {t}
            """.format(amount=self.amount, t=self.t)
        ).strip()

    @property
    def caperiod_present_ref(self) -> dt.date:
        from fred.utils.dateops import datetime_utcnow
        return datetime_utcnow().date()

    @property
    def default_period_unit(self) -> CompoundingUnit:
        return CompoundingUnit.DAYS

    @staticmethod
    def struct_cls() -> type['XCFProjectStruct']:
        return XCFProjectStruct

    @staticmethod
    def struct(flows: Optional[list['XCF']] = None) -> 'XCFProjectStruct':
        return XCFProjectStruct(
            flows={cf.t: cf for cf in flows or []}
        )

    def caperiod_diff(self, end: dt.date) -> int:
        return (end - self.t).days
    
    def caperiod_add(self, periods: int) -> dt.date:
        return self.t + dt.timedelta(days=periods)

    def to_pcf(self, reference_date: Optional[dt.date] = None, as_current: bool = False) -> CashFlowInterface:
        from fred.ftb.cashflow._intpos import PCF
        reference_date = reference_date or self.caperiod_present_ref
        delta_days = 0 if as_current else (self.t - reference_date).days
        return PCF(
            amount=self.amount,
            t=delta_days
        )


@dataclass(slots=True, frozen=True)
class XCFProjectStruct(ProjectStructInterface[dt.date]):
    flows: dict[dt.date, XCF] = field(default_factory=dict)

    def to_pcf(self, start_date: Optional[dt.date | str] = None, today: bool = False) -> ProjectStructInterface[int]:
        from fred.ftb.cashflow._intpos import PCFProjectStruct
        from fred.utils.dateops import datetime_utcnow
        base = start_date or (
            datetime_utcnow().date() if today
            else min(self.flows.keys(), default=datetime_utcnow().date())
        )
        if isinstance(base, str):
            base = dt.date.fromisoformat(base)
        struct = PCFProjectStruct.empty()
        for cf in self.flows.values():
            struct = struct.add_cashflow(cf.to_pcf(reference_date=base, as_current=False))
        return struct