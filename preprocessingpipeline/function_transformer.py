from sklearn.preprocessing import FunctionTransformer as FT  # type: ignore


def identity(x):
    return x


class FunctionTransformer(FT):
    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x, y)

    def transform(self, x, y):
        x = self._check_input(x, reset=False)
        return self._transform(x, y, func=self.func, kw_args=self.kw_args)

    def _transform(self, x, y, func=None, kw_args=None):
        if func is None:
            func = identity

        self._x_hat, self._y_hat = func(x, y, **(kw_args if kw_args else {}))
        return self._x_hat
