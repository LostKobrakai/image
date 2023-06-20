if match?({:module, _module}, Code.ensure_compiled(Nx)) do
  defmodule Image.Nx do
    import Nx
    import Nx.Defn.Kernel, only: [keyword!: 2]

    alias Nx.Tensor, as: T
    alias Vix.Vips.Image, as: Vimage

    @doc """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by $out_i = a_{i+1} - a_i$ along the given axis,
    higher differences are calculated by using `diff` recursively.

    ## Options

      * `:order` - the number of times to perform the difference. Defaults to `1`
      * `:axis` - the axis to perform the difference along. Defaults to `-1`

    ## Examples

        iex> Nx.diff(Nx.tensor([1, 2, 4, 7, 0]))
        #Nx.Tensor<
          s64[4]
          [1, 2, 3, -7]
        >

        iex> Nx.diff(Nx.tensor([1, 2, 4, 7, 0]), order: 2)
        #Nx.Tensor<
          s64[3]
          [1, 1, -10]
        >

        iex> Nx.diff(Nx.tensor([[1, 3, 6, 10], [0, 5, 6, 8]]))
        #Nx.Tensor<
          s64[2][3]
          [
            [2, 3, 4],
            [5, 1, 2]
          ]
        >

        iex> Nx.diff(Nx.tensor([[1, 3, 6, 10], [0, 5, 6, 8]]), axis: 0)
        #Nx.Tensor<
          s64[1][4]
          [
            [-1, 2, 0, -2]
          ]
        >

        iex> Nx.diff(Nx.tensor([1, 2, 4, 7, 0]), order: 0)
        #Nx.Tensor<
          s64[5]
          [1, 2, 4, 7, 0]
        >

        iex> Nx.diff(Nx.tensor([1, 2, 4, 7, 0]), order: -1)
        ** (ArgumentError) order must be non-negative but got: -1
    """
    @doc type: :ndim
    def diff(tensor, opts \\ []) do
      opts = keyword!(opts, order: 1, axis: -1)
      %T{shape: shape, names: names} = tensor = to_tensor(tensor)
      n = opts[:order]
      axis = Nx.Shape.normalize_axis(shape, opts[:axis], names)

      if rank(tensor) == 0 do
        raise ArgumentError, "cannot compute diff of a scalar"
      end

      if n < 0 do
        raise ArgumentError, "order must be non-negative but got: #{inspect(n)}"
      end

      axis_size = Nx.axis_size(tensor, axis)

      Enum.reduce(0..(n - 1)//1, tensor, fn x, acc ->
        subtract(
          slice_along_axis(acc, 1, axis_size - x - 1, axis: axis),
          slice_along_axis(acc, 0, axis_size - x - 1, axis: axis)
        )
      end)
    end

    @square_256 256 ** 2

    def unique_colors(%Vimage{} = image) do
      with {:ok, tensor} <- Image.to_nx(image) do
        colors_base256 =
          tensor
          |> Nx.multiply(Nx.tensor([[1, 256, @square_256]]))
          |> Nx.sum(axes: [2])
          |> Nx.flatten()
          |> Nx.sort()

        diff =
          diff(colors_base256)

        unique_indices_selector =
          Nx.concatenate([Nx.tensor([1]), Nx.not_equal(diff, 0)])

        marked_unique_indices =
          Nx.select(unique_indices_selector, Nx.iota(colors_base256.shape), -1)

        repeated_count =
          Nx.to_number(Nx.sum(Nx.logical_not(unique_indices_selector)))

        unique_indices =
          marked_unique_indices
          |> Nx.sort()
          |> Nx.slice_along_axis(repeated_count, Nx.size(marked_unique_indices) - repeated_count, axis: 0)

        collapsed_unique_colors =
          Nx.take(colors_base256, unique_indices)

        b = Nx.quotient(collapsed_unique_colors, @square_256)
        rem = Nx.remainder(collapsed_unique_colors, @square_256)
        g = Nx.quotient(rem, 256)
        r = Nx.remainder(rem, 256)

        unique_colors = Nx.stack([r, g, b], axis: 1)
        count = div(Nx.size(colors_base256), 3)
        max = Nx.to_number(Nx.reduce_max(unique_indices))
        color_count = Nx.concatenate([diff(unique_indices), Nx.tensor([count - max])])

        Nx.to_list(color_count)
        |> Enum.zip(Nx.to_list(unique_colors))
        |> Enum.sort(:desc)
      end
    end

    def kmeans(%Vimage{} = image, options \\ []) do
      kmeans =
        image
        |> unique_colors()
        |> Scholar.Cluster.KMeans.fit(options)

        kmeans.clusters
        |> Nx.round()
        |> Nx.as_type(:u8)
    end
  end
end
