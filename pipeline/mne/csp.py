import mne  # type: ignore
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing import Dict, List, Literal, Optional, Tuple, Union, cast


class NDCSP(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 4,
        reg: Optional[Union[float, str]] = None,
        log: Optional[bool] = None,
        cov_est: Literal["concat", "epoch"] = "concat",
        transform_into: Literal["average_power", "csp_space"] = "average_power",
        norm_trace: bool = False,
        cov_method_params: Optional[Dict] = None,
        rank: Optional[Union[Dict, Literal["info", "full"]]] = None,
        component_order: Literal["mutual_info", "alternate"] = "mutual_info",
        full_data: Optional[Literal["partial", "full"]] = None,
        fbank_axis: Optional[int] = None,
        axes: Optional[List] = None,
    ):
        """Added an axes parameter in case the input has more than three dimensions,
        in which case the channel and time dimensions must be listed in the
        axes parameter so that the data can be appropriately shaped.

        Parameters
        ----------
        full_data : None, "Train", "Held out"
            In the event that a masked array is passed, when full_data is not none
            all masked data is processed. If "full", then csps are fitted with
            all data. If "partial", then masked data is only transformed and
            does *not* contribute to fitting
        fbank_axis : Optional[int]
            In the event that a filter bank step preceeds CSP (e.g., for Filter
            Bank CSP), this parameter identifies the index of the dimension
            where the filters are located. None, indicates not filter bank was
            processed.
        axes : Optional[List]
            In the event that the input data is expected to be multidimensional
            (> 3), these axes provide the option to specify where the channel
            and time dimensions should be expected within the shape of the
            incoming matrix data.

        For other parameters, see mne.decoding.CSP
        """
        self.n_components: int = n_components
        self.reg: Optional[Union[float, str]] = reg
        self.log: Optional[bool] = log
        self.cov_est: Literal["concat", "epoch"] = cov_est
        self.transform_into: Literal["average_power", "csp_space"] = transform_into
        self.norm_trace: bool = norm_trace
        self.cov_method_params: Optional[Dict] = cov_method_params
        self.rank: Optional[Union[Dict, Literal["info", "full"]]] = rank
        self.component_order: Literal["mutual_info", "alternate"] = component_order
        self.full_data: Optional[Literal["partial", "full"]] = full_data
        self.fbank_axis: Optional[int] = fbank_axis
        self.axes: List[int] = [-2, -1] if axes is None else axes
        self.csps: List[BaseEstimator] = []
        self._removed_indices: np.ndarray = np.empty(0)
        self._removed_data: np.ndarray = np.empty(0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NDCSP":
        """Fit the data to one or more CSP transformers. There are as many CSP
        transformers as there are filter banks within the data

        Parameters
        ----------
        x : np.ndarray
            The input data
        y : np.ndarray
            The labels

        Returns
        -------
        NDCSP
            The fitted transformer
        """

        # Initialize n CSP instances for each filter bank
        self.init_csps(x)
        x, yt = self.prep_input(x, y)
        y = cast(np.ndarray, yt)
        for i, csp in zip(x, self.csps):
            csp.fit(i, y)
        return self

    def fit_transform(self, x: np.ndarray, y: np.ndarray, **fit_params) -> np.ndarray:
        """Fit the data and then transform it

        Parameters
        ----------
        x : np.ndarray
            The input data
        y : np.ndarray
            The labels

        Returns
        -------
        np.ndarray
            The transformed data
        """
        x = self.fit(x, y).transform(x)
        return x

    def init_csps(self, x: np.ndarray) -> None:
        """Initialize 1 or more CSP transformers. One for every filter bank available.

        Parameters
        ----------
        x : np.ndarray
            The input data to the transformer
        """
        self.n_fbanks: int = 1 if self.fbank_axis is None else x.shape[self.fbank_axis]
        self.csps = [
            mne.decoding.CSP(
                self.n_components,
                self.reg,
                self.log,
                self.cov_est,
                self.transform_into,
                self.norm_trace,
                self.cov_method_params,
                self.rank,
                self.component_order,
            )
        ] * self.n_fbanks

    def prep_input(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare input data to be formated for CSP processing

        Parameters
        ----------
        x : np.ndarray
            The input data to be preprocessed
        y : Optional[np.ndarray]
            The labels associoated with the input

        Returns
        -------
        xt : np.ndarray
        yt : np.ndarray
        """
        self._removed_indices = np.empty(0)
        x, y = self.prep_input_shape(x, y)

        if isinstance(x, np.ma.core.MaskedArray):
            x, y = self.prep_input_masked(
                cast(np.ma.core.MaskedArray, x), cast(np.ma.core.MaskedArray, y)
            )

        return x, y

    def prep_input_ct(self, x: np.ndarray, ndim: int) -> np.ndarray:
        """Prepare input by ensuring that the channel and time dimensions are
        placed at the end of the sequence for proper analyssi

        Parameters
        ----------
        x : np.ndarray
            input data
        ndim : int
            The original number of dimensions of x

        Returns
        -------
        np.ndarray
            Array with moved channel and time axes
        """
        c_axis, t_axis = self.relative_axes(ndim)
        x = np.moveaxis(x, [c_axis, t_axis], [-2, -1])
        return x

    def prep_input_f(self, x) -> np.ndarray:
        """Prepare the input array for filter bank processing. If there is a
        filter bank axis, it is placed in the front dimension for looping
        during transformation

        Parameters
        ----------
        x : np.ndarray
            The input data, typically returned from `prep_input_ct`

        Returns
        -------
        np.ndarray
            Adjusted input
        """
        if self.fbank_axis is None:
            x = x[np.newaxis, ...]
        else:
            f_axis: int = self.fbank_axis
            x = np.moveaxis(x, f_axis, 0)
        return x

    def prep_input_flat(self, x: np.ndarray, input_shape: Tuple) -> np.ndarray:
        """Prepare the input data for CSP transformation by flattening all axes
        not related to channel data, time data, or filter bank data

        Parameters
        ----------
        x : np.ndarray
            The input data. Typically processed after `prep_input_f`
        input_shape : Tuple
            The original input shape of x prior to any preparation

        Returns
        -------
        np.ndarray
            The transformed data
        """
        c_axis, t_axis = self.relative_axes(len(input_shape))
        f: int = x.shape[0]
        c: int = input_shape[c_axis]
        t: int = input_shape[t_axis]
        return x.reshape(f, -1, c, t)

    def prep_input_ma(
        self, x: np.ma.core.MaskedArray, y: Optional[np.ma.core.MaskedArray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """This function ensures that the final preparation of the data results
        in np.ndarrays that are not masked arrays.

        Definitions
        -----------
        to mask: The process of applying a mask to an array
        to unmask: Simply removing the mask of a masked array
        to demask: Removing the masked elements from a masked array
        to remask: To refill elements of a demasked list

        Parameters
        x : np.core.ma.MaskedArray
            The masked array to be unmasked
        y : np.core.ma.MaskedArray
            The masked set of labeles to be unmasked

        Returns
        -------
        xa : np.ndarray
            The unmasked x data
        ya : np.ndarray
            The unmasked y data
        """
        # input data must be 4D, processed through `prep_input_shape`
        assert (
            x.ndim == 4
        ), f"the unmasking process expects 4 dimensions but received, {x.ndim}"

        # NOTE: We currently assume that if `y` is present than the data is
        # being prepared for fitting. If `y` is absent (i.e., None), the
        # process assumes trasnformation
        if y is not None:
            yreq: np.ma.core.MaskedArray = cast(np.ma.core.MaskedArray, y)
            if self.full_data == "full":
                # Simply unmask the data
                ya: Optional[np.ndarray] = yreq.data
                xa: np.ndarray = x.data
            else:
                # Demask the data. This is destructive in that elements are removed
                # The current assumpion is that the mask only applies to
                # windows/epochs (2nd dimension). If there are other ragged
                # factors such as different channels or times, interpolation
                # must be done but this should be done prior, as this class is
                # not responsible for that.
                ya = yreq.data[~yreq.mask]
                x_shape: List = list(x.shape)
                x_shape[1] = cast(np.ndarray, ya).size
                xa = x.data[~x.mask].reshape(*x_shape)

                # stored for later reconstruction
                self._removed_indices = np.where(yreq.flatten().mask)[0]
        else:
            f, w, c, t = x.shape
            if self.full_data is not None:
                # simply unmask the data
                xa = x.data
            else:
                # demask the data
                xa = x.data[~x.mask].reshape(f, -1, c, t)
                self._removed_indices = np.where(x.mask[0, :, 0, 0])[0]
            ya = y

        return xa, ya

    def prep_input_shape(
        self, x: np.ndarray, y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """The purpose of this preparation step is to assert that the data
        being fed into the CSP transformer is a filter banked list of 3D data
        sets. The final output should be of the following shape:
            (filter_banks, epochs, channels, time)

        Parameters
        ----------
        x : np.ndarray
            The input data to prepare
        y : np.ndarray
            The associated labels for fitting

        Returns
        -------
        x : np.ndarray
            The restructured data
        y : np.ndarray
            The flattened label vector
        """
        input_shape: Tuple = x.shape
        if x.ndim == 3:
            assert self.fbank_axis is None
            x = x[np.newaxis, ...]
        elif x.ndim > 3:
            x = self.prep_input_ct(x, len(input_shape))
            x = self.prep_input_f(x)
            x = self.prep_input_flat(x, input_shape)

        if y is not None and y.ndim > 1:
            y = y.flatten()
        if y is not None:
            assert (
                x.shape[1] == y.shape[0]
            ), f"WARNING: x shape: {x.shape}\ny shape: {y.shape}"

        assert (
            x.ndim == 4
        ), f"the output data was expected to be 4 dimensions but is {x.ndim}"
        return x, y

    def prep_input_masked(
        self, x: np.ma.core.MaskedArray, y: Optional[np.ma.core.MaskedArray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """This preparation step ensures that the data being fed to the CSP fit
        and transform operations is unmasked. Do this provides flexibility for
        processing data that should not fully be considered. Two examples of
        this are altered labeling schemes and for ragged arrays with mixed
        lengths. NOTE: at present this operation only considered the labelling
        scheme scenario

        Parameters
        ----------
        x : np.ndarray
            The input data
        y : Optional[np.ndarray]
            The labels

        Returns
        -------
        x : np.ndarray
        y : np.ndarray
        """
        # In some cases the mask is all False and can simply be removed
        if not isinstance(x.mask, np.ndarray) or (~x.mask).all():
            x = x.data
            y = y if y is None else y.data
            return x, y

        x_out: np.ndarray
        y_out: Optional[np.ndarray]
        x_out, y_out = self.prep_input_ma(x, y)
        return x_out, y_out

    def prep_output(
        self, x: np.ndarray, input_shape: Tuple, mask: Optional[np.ndarray] = None
    ):
        """Prepare the output data for return to pipeline after CSP
        processing

        Parameters
        ----------
        x : np.ndarray
            The data after processing through CSP
        input_shape : Tuple
            The original shape of the input data to this estimator

        Returns
        -------
        np.ndarray
            The adjust output
        """
        if mask is not None:
            x = self.prep_output_masked(x, mask)

        x = self.prep_output_shape(x, input_shape)

        return x

    def prep_output_ct(self, x: np.ndarray, ndim: int) -> np.ndarray:
        """Replace the channel and time axes back to their original
        positions

        Parameters
        ----------
        x : np.ndarray
            The input data after processing through `prep_output_f`
        ndim : int
            The number of dimensions of the original data passed to the estimator

        Returns
        -------
        np.ndarray
            The output data with channel and time axes restored to their
            original positions
        """
        c_axis, t_axis = self.relative_axes(ndim)
        if self.transform_into == "average_power":
            return np.moveaxis(x, -1, c_axis)
        else:
            return np.moveaxis(x, [-2, -1], [c_axis, t_axis])

    def prep_output_f(self, x: np.ndarray) -> np.ndarray:
        """Replace the filte bank axis back to its original position

        Parameters
        ----------
        x : np.ndarray
            The input data after processing through `prep_output_flat`

        Returns
        -------
        np.ndarray
            The output data with filter bank axis restored
        """
        if self.fbank_axis is None:
            return x[0]
        else:
            return np.moveaxis(x, 0, self.fbank_axis)

    def prep_output_flat(self, x: np.ndarray, input_shape: Tuple) -> np.ndarray:
        """Prepare the output data by restoring the axes lost to flattening

        Parameters
        ----------
        x : np.ndarray
            The input data
        input_shape : Tuple
            The shape of the original data passed to `fit` or `transform`

        Returns
        -------
        np.ndarray
            The processed output data with all axes restored
        """
        intermediate_shape: List = self.prep_output_inter(input_shape)
        return x.reshape(intermediate_shape)

    def prep_output_inter(self, input_shape: Tuple) -> List:
        """Compute the appropriate intermediate shape of the data following CSP
        transformation to restore axes that were lost to flattening during
        preparation

        Parameters
        ----------
        input_shape : Tuple
            The original shape of the input data to `fit` or `transform` functions

        Returns
        -------
        List
            The estimated intermediate shape of the data immediately following
            CSP transformation
        """
        # Compute intermediate shape
        ndim: int = len(input_shape)
        if ndim == 3:
            result: List[int] = [1] + list(input_shape)
        else:
            d: np.ndarray = np.ones(input_shape)
            d = self.prep_input_ct(d, ndim)
            d = self.prep_input_f(d)
            result = list(d.shape)

        # Replace n_channels with n_components
        c: int = result[-2]
        result[-2] = self.n_components if self.n_components < c else c

        # account for transformation rule
        if self.transform_into == "average_power":
            result = result[:-1]
        return result

    def prep_output_masked(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """This method functions as the inverse opperation of
        `prep_input_masked` and `prep_input_ma`

        Parameters
        ----------
        x : np.ndarray
            The transformed data of shape (filters, epochs, csp) OR (filters,
            epochs, csp, time)

        Returns
        -------
        np.ndarray
            The masked or re-masked array data
        """
        if self.full_data is None:
            x = self.prep_output_remask(x)
        else:
            # Adjust the mask
            mask, _ = self.prep_input_shape(mask, None)
            if self.transform_into == "csp_space":
                f, e, c, t = x.shape
                mask = mask[:, :, :c, ...]
            else:
                f, e, c = x.shape
                mask = mask[:, :, :c, ...].any(axis=-1)

            x = np.ma.MaskedArray(x, mask=mask)

        return x

    def prep_output_remask(self, x: np.ndarray) -> np.ma.core.MaskedArray:
        """Remasks a data that previously had it's data stripped. This ensures
        that the output size matches. Holes are filled with NaN

        Parameters
        ----------
        x : np.ndarray
            The transformed input data

        Returns
        -------
        np.ma.core.MaskedArray
            The remasked data
        """
        if self.transform_into == "average_power":
            f, w, c = x.shape
            insert: np.ndarray = np.ones((c)) * np.nan
        elif self.transform_into == "csp_space":
            f, w, c, t = x.shape
            insert = np.ones((c, t)) * np.nan

        indicies: np.ndarray = self._removed_indices - np.arange(
            self._removed_indices.size
        )
        x = np.insert(x, indicies, insert, axis=1)
        return np.ma.masked_invalid(x)

    def prep_output_shape(self, x: np.ndarray, input_shape: Tuple) -> np.ndarray:
        ndim: int = len(input_shape)
        if ndim == 3:
            assert self.fbank_axis is None
            x = x[0]
        elif ndim > 3:
            x = self.prep_output_flat(x, input_shape)
            x = self.prep_output_f(x)
            x = self.prep_output_ct(x, ndim)

        return x

    def relative_axes(self, ndim: int) -> List[int]:
        """Compute the axes relative to the number of dimensions of the input
        data

        Parameters
        ----------
        ndim : int
            The number of dimensions of the input data

        Returns
        -------
        List[int]
            The channel and time axis relative to the dimensions of the input
            data
        """
        return [self.resolve_pos_axis(i, ndim) for i in self.axes]

    def resolve_pos_axis(self, axis: int, ndims: int) -> int:
        """Given an integer to index  into an axis, return the positive
        variant. Numpy and lists allow for negative indexing. If the value is
        negative, this returns the positive location

        Parameters
        ----------
        axis : int
            The input axis
        ndims : int
            The total number of dimensions in the input data

        Returns
        -------
        int
            The positive value
        """
        if axis < 0:
            axis = ndims + axis

        return axis

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform x through CSP matricies

        Parameters
        ----------
        x : np.ndarray
            The input data

        Returns
        -------
        np.ndarray
            The transformed data
        """
        input_shape: Tuple = x.shape
        input_mask: Optional[np.ndarray] = (
            None if not isinstance(x, np.ma.core.MaskedArray) else x.mask
        )
        x, _ = self.prep_input(x)
        x = np.stack([csp.transform(i) for i, csp in zip(x, self.csps)])
        x = self.prep_output(x, input_shape, input_mask)
        return x
