"""Decoder normalization and residual equations shared by ordinary and cached execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar, assert_never

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from stackformers.feedforward.protocols import FeedForward
    from stackformers.norm.protocols import Norm

from stackformers.sequence import SequenceInput

InputT = TypeVar("InputT", bound=SequenceInput)
ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")


class NoContext(NamedTuple):
    """The caller omitted context, so the entire cross-attention branch is skipped."""


class WithContext(NamedTuple, Generic[ContextT]):
    """A complete context value required by the supplied cross-attention operation."""

    value: ContextT


def pre_norm_decoder(
    x_input: InputT,
    self_attn: Callable[[InputT], tuple[Tensor, StateT]],
    cross_attn: Callable[[InputT, ContextT], Tensor],
    context: NoContext | WithContext[ContextT],
    ff: FeedForward,
    norm_self: Norm,
    norm_cross: Norm,
    norm_ff: Norm,
) -> tuple[InputT, StateT]:
    """Apply the pre-norm equation and carry self-attention state unchanged."""
    normalized_self = norm_self(x_input.x)
    self_input = x_input._replace(x=normalized_self)
    self_out, state = self_attn(self_input)
    x = x_input.x + self_out

    x_input = x_input._replace(x=x)
    match context:
        case WithContext(value=ctx_input):
            normalized_cross = norm_cross(x_input.x)
            cross_input = x_input._replace(x=normalized_cross)
            cross_out = cross_attn(cross_input, ctx_input)
            x = x + cross_out
        case NoContext():
            pass
        case _:
            assert_never(context)

    normalized_ff = norm_ff(x)
    ff_out = ff(normalized_ff)
    x = x + ff_out
    return x_input._replace(x=x), state


def post_norm_decoder(
    x_input: InputT,
    self_attn: Callable[[InputT], tuple[Tensor, StateT]],
    cross_attn: Callable[[InputT, ContextT], Tensor],
    context: NoContext | WithContext[ContextT],
    ff: FeedForward,
    norm_self: Norm,
    norm_cross: Norm,
    norm_ff: Norm,
) -> tuple[InputT, StateT]:
    """Apply the post-norm equation and carry self-attention state unchanged."""
    self_out, state = self_attn(x_input)
    x = x_input.x + self_out
    x = norm_self(x)

    x_input = x_input._replace(x=x)
    match context:
        case WithContext(value=ctx_input):
            cross_out = cross_attn(x_input, ctx_input)
            x = x + cross_out
            x = norm_cross(x)
        case NoContext():
            pass
        case _:
            assert_never(context)

    ff_out = ff(x)
    x = x + ff_out
    x = norm_ff(x)
    return x_input._replace(x=x), state


def sandwich_norm_decoder(
    x_input: InputT,
    self_attn: Callable[[InputT], tuple[Tensor, StateT]],
    cross_attn: Callable[[InputT, ContextT], Tensor],
    context: NoContext | WithContext[ContextT],
    ff: FeedForward,
    norm_self_pre: Norm,
    norm_self_post: Norm,
    norm_cross_pre: Norm,
    norm_cross_post: Norm,
    norm_ff_pre: Norm,
    norm_ff_post: Norm,
) -> tuple[InputT, StateT]:
    """Apply the sandwich-norm equation and carry self-attention state unchanged."""
    normalized_self = norm_self_pre(x_input.x)
    self_input = x_input._replace(x=normalized_self)
    self_out, state = self_attn(self_input)
    self_branch = norm_self_post(self_out)
    x = x_input.x + self_branch

    x_input = x_input._replace(x=x)
    match context:
        case WithContext(value=ctx_input):
            normalized_cross = norm_cross_pre(x_input.x)
            cross_input = x_input._replace(x=normalized_cross)
            cross_out = cross_attn(cross_input, ctx_input)
            cross_branch = norm_cross_post(cross_out)
            x = x + cross_branch
        case NoContext():
            pass
        case _:
            assert_never(context)

    normalized_ff = norm_ff_pre(x)
    ff_out = ff(normalized_ff)
    ff_branch = norm_ff_post(ff_out)
    x = x + ff_branch
    return x_input._replace(x=x), state


def reordered_norm_decoder(
    x_input: InputT,
    self_attn: Callable[[InputT], tuple[Tensor, StateT]],
    cross_attn: Callable[[InputT, ContextT], Tensor],
    context: NoContext | WithContext[ContextT],
    ff: FeedForward,
    norm_self: Norm,
    norm_cross: Norm,
    norm_ff: Norm,
) -> tuple[InputT, StateT]:
    """Apply the reordered-norm equation and carry self-attention state unchanged."""
    self_out, state = self_attn(x_input)
    self_branch = norm_self(self_out)
    x = x_input.x + self_branch

    x_input = x_input._replace(x=x)
    match context:
        case WithContext(value=ctx_input):
            cross_out = cross_attn(x_input, ctx_input)
            cross_branch = norm_cross(cross_out)
            x = x + cross_branch
        case NoContext():
            pass
        case _:
            assert_never(context)

    ff_out = ff(x)
    ff_branch = norm_ff(ff_out)
    x = x + ff_branch
    return x_input._replace(x=x), state
