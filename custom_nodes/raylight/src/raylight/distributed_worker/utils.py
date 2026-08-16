import torch
import torch.distributed as dist
import functools
from ray.experimental import tqdm_ray as ray_tqdm_module
from ray.experimental.tqdm_ray import tqdm as ray_tqdm
import tqdm.auto as tqdm_auto


_RAY_TQDM_ALLOWED_KWARGS = {"desc", "total", "unit", "position", "flush_interval_s"}


def _filter_ray_tqdm_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if key in _RAY_TQDM_ALLOWED_KWARGS}


class _RayTqdmCompat:
    def __init__(self, iterable=None, **kwargs):
        self._iterable = iterable
        self._bar = ray_tqdm(iterable, **_filter_ray_tqdm_kwargs(kwargs))

    def __iter__(self):
        try:
            yield from self._bar
            return
        except TypeError:
            pass

        if self._iterable is None:
            return

        for item in self._iterable:
            yield item
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getattr__(self, name):
        return getattr(self._bar, name)

    @property
    def n(self):
        return getattr(self._bar, "n", 0)

    @n.setter
    def n(self, value):
        try:
            self._bar.n = value
        except Exception:
            self.__dict__["_n"] = value

    @property
    def total(self):
        return getattr(self._bar, "total", None)

    @total.setter
    def total(self, value):
        try:
            self._bar.total = value
        except Exception:
            self.__dict__["_total"] = value

    def update(self, n=1):
        update = getattr(self._bar, "update", None)
        if update is None:
            return None
        return update(n)

    def refresh(self, *args, **kwargs):
        refresh = getattr(self._bar, "refresh", None)
        if refresh is None:
            return None
        try:
            return refresh(*args, **kwargs)
        except TypeError:
            return refresh()

    def close(self):
        close = getattr(self._bar, "close", None)
        if close is None:
            return None
        return close()

    def set_description(self, desc=None, refresh=True):
        set_description = getattr(self._bar, "set_description", None)
        if set_description is None:
            return None
        try:
            return set_description("" if desc is None else str(desc), refresh=refresh)
        except TypeError:
            return set_description("" if desc is None else str(desc))

    def set_description_str(self, desc=None, refresh=True):
        return self.set_description(desc, refresh=refresh)

    def set_postfix_str(self, s="", refresh=True):
        return self.set_description("" if s is None else str(s), refresh=refresh)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        values = {}
        if ordered_dict is not None:
            if hasattr(ordered_dict, "items"):
                values.update(ordered_dict)
            else:
                values.update(dict(ordered_dict))
        values.update(kwargs)
        postfix = ", ".join(f"{key}={value}" for key, value in values.items())
        return self.set_postfix_str(postfix, refresh=refresh)


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        import comfy.sample as comfy_sample

        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy_sample.prepare_noise(latent_image, self.seed, batch_inds)


# Monkey patch-unpatch tqdm and trange so it does not broke the progress bar
def _patch_ray_tqdm_close_once():
    if getattr(ray_tqdm_module, "_raylight_clear_on_close", False):
        return

    original_close = ray_tqdm_module._Bar.close

    def close_and_clear(self):
        try:
            self.bar.clear()
        except Exception:
            pass
        original_close(self)

    ray_tqdm_module._Bar.close = close_and_clear
    ray_tqdm_module._raylight_clear_on_close = True


def patch_ray_tqdm(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _patch_ray_tqdm_close_once()

        rank = dist.get_rank()
        orig_tqdm = tqdm_auto.tqdm
        orig_trange = tqdm_auto.trange
        if rank == 0:
            def ray_tqdm_absorb_disable(*a, **k):
                return _RayTqdmCompat(*a, **k)

            def ray_trange_absorb_disable(*a, **k):
                return _RayTqdmCompat(range(*a), **k)

            tqdm_auto.tqdm = ray_tqdm_absorb_disable
            tqdm_auto.trange = ray_trange_absorb_disable

        try:
            return fn(*args, **kwargs)
        finally:
            tqdm_auto.tqdm = orig_tqdm
            tqdm_auto.trange = orig_trange

    return wrapper
