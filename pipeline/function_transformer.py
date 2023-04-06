from sklearn.preprocessing import FunctionTransformer as FT


class FunctionTransformer(FT):
    def transform(self, x, y):
        x = self._check_input(x, reset=False)
        return self._transform(x, y, func=self.func, kw_args=self.kw_args)

    def _transform(self, x, y, func=None, kw_args=None):
        if func is None:
            func = lambda x: x

        self._x_hat, self._y_hat = func(x, y, **(kw_args if kw_args else {}))
        return self._x_hat
