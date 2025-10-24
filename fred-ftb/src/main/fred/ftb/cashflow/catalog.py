import enum
import datetime as dt
from typing import Optional

from fred.ftb.cashflow.interface import CashFlowInterface, ProjectStructInterface
from fred.ftb.cashflow._intpos import PCF
from fred.ftb.cashflow._datepos import XCF


class CashflowCatalog(enum.Enum):
    PCF = PCF
    XCF = XCF

    @classmethod
    def auto(cls, amount: float, t: int | str | dt.date | dt.datetime) -> CashFlowInterface:
        match t:
            case int():
                return PCF(amount=amount, t=t)
            case str():
                return XCF(amount=amount, t=dt.date.fromisoformat(t))
            case dt.date():
                return XCF(amount=amount, t=t)
            case dt.datetime():
                return XCF(amount=amount, t=t.date())
            case _:
                raise TypeError(f"Unsupported type for time period 't': {type(t)}")

    @property
    def ref(self) -> type[CashFlowInterface]:
        return self.value

    def struct(self, flows: Optional[list[CashFlowInterface]] = None) -> ProjectStructInterface:
        return self.value.struct(flows=flows or [])

    def __call__(self, *args, **kwargs) -> CashFlowInterface:
        return self.value(*args, **kwargs)