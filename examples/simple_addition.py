class Custom:

    def _check_params(self):
        # check if params is a dict and has the required keys
        if not isinstance(self.params, dict) or not all(
                key in self.params for key in ('graph_type', 'buffer_size', 'update_interval')):
            raise ValueError('params must be a dict and have the required keys')

    def __init__(self):

        # Params for graphs seeting
        # required keys: graph_type , update_interval
        self.params = dict(
            # 'line' or 'bar', required
            graph_type='line',
            # in ms, required
            update_interval=100,
            # in seconds, required
            buffer_size=5,
            # optional, but highly recommend to set
            # x_range=[0, 2500],
            # optional, but highly recommend to set
            # y_range=[-1e-4, 1e4],
            # optional, but highly recommend to set
            # x_ticks=[0, 500, 1000, 1500, 2000, 2500],
            # optional, but highly recommend to set
            # y_ticks=[0, 500, 1000, 1500, 2000, 2500],
            # optional, but highly recommend to set
            x_label='Time steps',
            # optional, but highly recommend to set
            y_label='Voltage plus one',
            # optional, but highly recommend to set
            title='Simple Addition',
        )

        self._check_params()

    def custom_func(self, data):
        """
        :param data: (1d numpy.array, or list), shape: (n, ), data feeding from GUI, n is the number of samples
        :return: (1d numpy.array), shape: (n, ), n is the number of dimension of the output
        """
        processed_data = data + 1

        return processed_data