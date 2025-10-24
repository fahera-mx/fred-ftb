from typing import TypeVar, Generic, Optional, Union

from fred.settings import logger_manager

from fred.ftb.rate import Rate
from fred.ftb.compounding.frequency import CompoundingFrequency
from fred.ftb.compounding.unit import CompoundingUnit


T = TypeVar('T')

logger = logger_manager.get_logger(name=__name__)


class CashFlowInterface(Generic[T]):
    amount: float
    t: T

    def caperiod_diff(self, end: T) -> int:
        """Return the capitalized period of the cash flow."""
        raise NotImplementedError("caperiod_diff method must be implemented by subclasses.")

    def caperiod_add(self, periods: int) -> T:
        """Add a number of periods to the cash flow's time period."""
        raise NotImplementedError("caperiod_add method must be implemented by subclasses.")

    @property
    def default_period_unit(self) -> CompoundingUnit:
        raise NotImplementedError("default_period_unit method must be implemented by subclasses.")

    def neutralize(self) -> 'CashFlowInterface[T]':
        """Return a neutral cash flow with zero amount."""
        return self.__class__(
            amount=0.0,
            t=self.t
        )

    def clone(self, amount: Optional[float] = None, t: Optional[T] = None) -> 'CashFlowInterface[T]':
        """Clone the cash flow with optional new amount and time period."""
        return self.__class__(
            amount=amount if amount is not None else self.amount,
            t=t if t is not None else self.t
        )

    @property
    def description(self) -> str:
        raise NotImplementedError("description method must be implemented by subclasses.")

    @property
    def caperiod_present_ref(self) -> T:
        raise NotImplementedError("caperiod_present_ref method must be implemented by subclasses.")

    @staticmethod
    def struct_cls() -> type['ProjectStructInterface[T]']:
        raise NotImplementedError("struct_cls method must be implemented by subclasses.")

    def shift(
            self,
            periods: int,
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
    ) -> 'CashFlowInterface[T]':
        """Shift the cash flow by a number of periods."""
        interest_rate = rate if isinstance(rate, Rate) else Rate.auto(value=rate, comp_freq=comp_freq, expr_unit=expr_unit)
        if comp_freq:
            interest_rate = interest_rate.freq_as(frequency=comp_freq)
        if expr_unit:
            interest_rate = interest_rate.expr_as(unit=expr_unit)
        return self.__class__(
            amount=interest_rate.apply(amount=self.amount, periods=periods, period_units=period_units or self.default_period_unit),
            t=self.caperiod_add(periods=periods)
        )

    def invert(self) -> 'CashFlowInterface[T]':
        """Invert the cash flow amount."""
        return self.__class__(
            amount=-self.amount,
            t=self.t
        )

    def to_present(
            self,
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
            present: Optional[T] = None
    ) -> 'CashFlowInterface[T]':
        """Calculate the present value of the cash flow given a discount rate."""
        return self.shift(
            periods=self.caperiod_diff(end=present or self.caperiod_present_ref),
            rate=rate,
            comp_freq=comp_freq,
            expr_unit=expr_unit,
            period_units=period_units,
        )

    def to_future(
            self,
            rate: float | Rate,
            n: T,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
    ) -> 'CashFlowInterface[T]':
        """Calculate the future value of the cash flow at time n given an interest rate."""
        return self.shift(
            periods=self.caperiod_diff(end=n),
            rate=rate,
            comp_freq=comp_freq,
            expr_unit=expr_unit,
            period_units=period_units,
        )

    def __add__(self, other: 'CashFlowInterface[T]') -> 'CashFlowInterface[T]':
        if self.t != other.t:
            raise ValueError("Cannot add cash flows with different time periods.")
        return self.__class__(
            amount=self.amount + other.amount,
            t=self.t
        )

    def __sub__(self, other: 'CashFlowInterface[T]') -> 'CashFlowInterface[T]':
        return self + other.invert()

    def combine(
            self,
            other: 'CashFlowInterface[T]',
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
    ) -> 'CashFlowInterface[T]':
        """Combine two cash flows occurring at different times into a single cash flow."""
        if self.t == other.t:
            return self + other
        # Shift the other cash flow to the time of this cash flow
        return self + other.shift(
            periods=other.caperiod_diff(end=self.t),
            rate=rate,
            comp_freq=comp_freq,
            expr_unit=expr_unit,
            period_units=period_units,
        )

    def __rshift__(self, other: Union['CashFlowInterface[T]', 'ProjectStructInterface[T]']) -> 'ProjectStructInterface[T]':
        struct = self.struct_cls().empty()
        struct = struct.register_cashflow(self)
        match other:
            case CashFlowInterface():
                return struct.register_cashflow(other)
            case ProjectStructInterface():
                return struct.merge(other)
            case _:
                raise TypeError(f"Unsupported type for combining CashFlowInterface: {type(other)}")
        return struct


class ProjectStructInterface[T]:
    flows: dict[T, CashFlowInterface]

    @classmethod
    def empty(cls) -> 'ProjectStructInterface':
        return cls(flows={})

    def clone(self) -> 'ProjectStructInterface[T]':
        return self.__class__(
            flows={t: cf.clone() for t, cf in self.flows.items()}
        )

    def as_list(self) -> list[CashFlowInterface[T]]:
        if not self.flows:
            return []
        init_position = min(self.flows.keys())
        last_position = max(self.flows.keys())
        size = self.flows[init_position].caperiod_diff(end=last_position) + 1
        out = [0] * size
        for cf in self.flows.values():
            position = -1 * cf.caperiod_diff(end=init_position)
            out[position] = cf.amount
        return out

    def __rshift__(self, other: Union['CashFlowInterface[T]', 'ProjectStructInterface[T]']) -> 'ProjectStructInterface[T]':
        struct = self
        match other:
            case CashFlowInterface():
                return struct.register_cashflow(other)
            case ProjectStructInterface():
                return struct.merge(other)
            case _:
                raise TypeError(f"Unsupported type for combining ProjectStructInterface: {type(other)}")

    def register_cashflow(self, cf: CashFlowInterface[T]) -> 'ProjectStructInterface[T]':
        self.flows[cf.t] = self.flows.pop(cf.t, cf.neutralize()) + cf
        return self
    
    def register_value(self, amount: float, t: T) -> 'ProjectStructInterface[T]':
        from fred.ftb.cashflow.catalog import CashflowCatalog as CFC
        cf = CFC.auto(amount=amount, t=t)
        return self.register_cashflow(cf)

    def register(
            self,
            amount: Optional[float] = None,
            t: Optional[T] = None,
            cf: Optional[CashFlowInterface[T]] = None
    ) -> 'ProjectStructInterface[T]':
        match (cf, amount, t):
            case (CashFlowInterface(), _, _):
                return self.register_cashflow(cf)
            case (_, float(), T()):
                return self.register_value(amount=amount, t=t)
            case _:
                raise ValueError("Either 'cf' or both 'amount' and 't' must be provided.")

    def invert(self) -> 'ProjectStructInterface':
        for t, cf in self.flows.items():
            self.flows[t] = cf.invert()
        return self

    def merge(self, other: 'ProjectStructInterface') -> 'ProjectStructInterface':
        for cf in other.flows.values():
            self.register(cf)
        return self

    def __add__(self, other: 'ProjectStructInterface') -> 'ProjectStructInterface':
        return self.merge(other)

    def __sub__(self, other: 'ProjectStructInterface') -> 'ProjectStructInterface':
        return self + other.invert()

    def profitable_at_rate(
            self,
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
            use_irr: bool = False,
    ) -> bool:
        if use_irr:
            logger.warning("Using IRR to determine profitability; this may be computationally intensive and doesn't guarantee a solution...")
            return self.irr() >= rate
        npv = self.npv(
            rate=rate,
            comp_freq=comp_freq,
            expr_unit=expr_unit,
            period_units=period_units,
        )
        return npv >= 0

    def profitable_at_exit(
            self,
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
            amount: Optional[float] = None,
            n: Optional[T] = None,
    ) -> bool:
        # Early exit if the total cash flows are insufficient
        if sum(self.as_list()) + (amount or 0) < 0:
            return False
        if amount is None:
            return self.profitable_at_rate(
                rate=rate,
                comp_freq=comp_freq,
                expr_unit=expr_unit,
                period_units=period_units,
                use_irr=False,
            )
        last = self.flows[max(self.flows.keys())]
        periods = last.caperiod_diff(end=n or last.t)
        if periods < 0:
            raise ValueError("Exit time period must be greater than or equal to the last cash flow time period.")
        struct = self.clone()
        exit_cf = last.clone(amount=amount, t=last.caperiod_add(periods=periods))
        return struct.register_cashflow(exit_cf).profitable_at_rate(
            rate=rate,
            comp_freq=comp_freq,
            expr_unit=expr_unit,
            period_units=period_units,
            use_irr=False
        )

    def npv(
            self,
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
            present: Optional[T] = None,
    ) -> float:
        return sum(
            cf.to_present(
                rate=rate,
                present=present,
                comp_freq=comp_freq,
                expr_unit=expr_unit,
                period_units=period_units,
            ).amount
            for cf in self.flows.values()
        )

    def irr(self, pyfloat_coerce: bool = False, raw: bool = False, unit: Optional[CompoundingUnit] = None) -> Optional[float | Rate]:
        if not raw:
            rate = self.irr(pyfloat_coerce=pyfloat_coerce, raw=True)
            if not rate:
                return None
            from fred.ftb.cashflow._datepos import XCFProjectStruct
            from fred.ftb.cashflow._intpos import PCFProjectStruct
            match self:
                # When dealing with date-based cash flows, default to daily compounding if no unit is provided
                case XCFProjectStruct():  # if only... case ProjectStructInterface[dt.date]():
                    unit = unit or CompoundingUnit.DAYS
                # When dealing with integer time periods, ensure a compounding unit is provided since the periods are ambiguous
                case PCFProjectStruct():  # if only... case ProjectStructInterface[int]():
                    if unit is None:
                        logger.warning("CompoundingUnit must be provided for integer time periods when raw=False... Using annual compounding by default.")
                        unit = CompoundingUnit.YEARS
                case _:
                    raise TypeError(f"Unsupported type for time period 'T': {T}")
            return Rate.auto(value=rate, comp_freq=unit.as_freq(), expr_unit=unit).expr_annual()
        if pyfloat_coerce:
            return float(self.irr(pyfloat_coerce=False))
        import numpy as np
        # Amounts in reverse order to represent the polynomial coefficients (from highest degree to constant term)
        cashflow_amounts = self.as_list()[::-1]
        # Get the polynomial roots
        roots = np.roots(cashflow_amounts)
        # Find real roots only (ignore complex or imaginary roots)
        realonly = (roots.imag == 0)
        if not realonly.any():
            return
        solutions = roots[realonly].real
        # Transform solution back to interest-rate representation
        rates = 1 / solutions - 1
        if len(rates) > 1:
            logger.debug(f"Multiple IRR solutions found: {rates}")
            logger.debug("Returning the closest-to-zero IRR available (starting from positive rates).")
        # Return the smallest positive rate, or the largest negative rate if no positive rates exist
        positiveonly = rates > 0
        return min(rates[positiveonly]) if positiveonly.any() else max(rates)
        
    def value_at(
            self,
            n: T,
            rate: float | Rate,
            comp_freq: Optional[CompoundingFrequency] = None,
            expr_unit: Optional[CompoundingUnit] = None,
            period_units: Optional[CompoundingUnit] = None,
    ) -> float:
        return sum(
            cf.to_future(
                rate=rate,
                comp_freq=comp_freq,
                expr_unit=expr_unit,
                period_units=period_units,
                n=n
            ).amount
            for cf in self.flows.values()
        )

    def get_figure(
            self,
            title: Optional[str] = None,
            disable_grid: bool = False,
            disable_irr: bool = False,
            raw: bool = False,
            unit: Optional[CompoundingUnit] = None,
            **kwargs
    ) -> 'plt.Figure':
        import matplotlib.pyplot as plt

        times = sorted(self.flows.keys())
        amounts = [self.flows[t].amount for t in times]

        if not title:
            title = "Cash Flow Over Time"
            irr = None if disable_irr else self.irr(raw=raw, unit=unit)
            if irr:
                title += f" (IRR: {irr.percent()}%)"

        # Allow caller to override color via kwargs['color']
        user_color = kwargs.pop('color', None)

        if user_color is None:
            # Elegant, muted green and deep red
            positive_color = '#2E7D32'  # muted green
            negative_color = '#C62828'  # deep red
            colors = [positive_color if a >= 0 else negative_color for a in amounts]
        else:
            colors = user_color

        fig, ax = plt.subplots()
        # Draw bars with specified colors; keep bars slightly opaque and with a subtle edge
        ax.bar(times, amounts, color=colors, zorder=2, edgecolor='black', linewidth=0.6, alpha=0.95, **kwargs)

        # More visible horizontal line at y=0
        ax.axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=5)
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Cash Flow Amount")
        ax.set_title(title)
        ax.grid(not disable_grid)
        return fig

    def plot(
            self,
            title: Optional[str] = None,
            disable_grid: bool = False,
            disable_irr: bool = False,
            raw: bool = False,
            unit: Optional[CompoundingUnit] = None,
            **kwargs,
    ) -> None:
        import matplotlib.pyplot as plt

        self.get_figure(
            title=title,
            disable_grid=disable_grid,
            disable_irr=disable_irr,
            raw=raw,
            unit=unit,
            **kwargs
        )
        plt.show()
