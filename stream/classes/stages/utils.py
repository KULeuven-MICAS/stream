def set_required_kwargs(required_kwargs, obj):
    """
    Sets the required keyword arguments as object attributes.
    Raises a ValueError if one of the required kwargs is not present in kwargs.

    Args:
        required_kwargs (list): The required keyword arguments as a list of strings.
        obj (Object): The object (typically the Stage) to set the attributes in.

    Raises:
        ValueError: When one of the required kwargs is not present in obj.kwargs.
    """
    for required_kwarg in required_kwargs:
        try:
            setattr(obj, required_kwarg, obj.kwargs[required_kwarg])
        except KeyError:
            raise ValueError(
                f'Please define "{required_kwarg}" for {obj.__class__.__name__}.'
            )
