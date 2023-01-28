defmodule Image.TestSupport do
  import ExUnit.Assertions
  alias Vix.Vips.Image, as: Vimage

  @images_path Path.join(__DIR__, "images")
  @validate_path Path.join(__DIR__, "validate")
  @acceptible_similarity 1.1

  def assert_files_equal(expected, result) do
    assert File.read!(expected) == File.read!(result)
  end

  def assert_images_equal(%Vimage{} = calculated_image, validate) when is_binary(validate) do
    validate_image = Image.open!(validate, access: :random)
    compare_images(calculated_image, validate_image)
  end

  def assert_images_equal(calculated, validate)
      when is_binary(calculated) and is_binary(validate) do
    validate_image = Image.open!(validate, access: :random)
    calculated_image = Image.open!(calculated, access: :random)

    compare_images(calculated_image, validate_image)
  end

  def assert_images_equal(%Vimage{} = calculated, %Vimage{} = validate) do
    compare_images(calculated, validate)
  end

  def image_path(name) do
    Path.join(@images_path, name)
  end

  def validate_path(name) do
    Path.join(@validate_path, name)
  end

  # From: https://github.com/libvips/libvips/discussions/2232
  # Calculate a single number for the match between two images, calculate the sum
  # of squares of differences,
  def compare_images(calculated_image, validate_image) do
    alias Image.Math

    {calculated_image, validate_image} =
      if Vimage.format(calculated_image) == Vimage.format(validate_image) do
        {calculated_image, validate_image}
      else
        {
          Vix.Vips.Operation.cast!(calculated_image, :VIPS_FORMAT_UCHAR),
          Vix.Vips.Operation.cast!(validate_image, :VIPS_FORMAT_UCHAR)
        }
      end

    similarity =
      calculated_image
      |> Math.subtract!(validate_image)
      |> Math.pow!(2)
      |> Vix.Vips.Operation.avg!()

    if similarity < @acceptible_similarity do
      assert true
    else
      path =
        validate_image
        |> Image.filename()
        |> String.replace("validate", "did_not_match")

      comparison_image =
        Vix.Vips.Operation.relational!(
          calculated_image,
          validate_image,
          :VIPS_OPERATION_RELATIONAL_EQUAL
        )

      Image.write!(comparison_image, path)

      flunk(
        "Calculated image did not match pre-existing validation image. " <>
          "Similarity score was #{inspect(similarity)}. " <>
          "See the image at #{path} for the image diff."
      )
    end
  end
end
