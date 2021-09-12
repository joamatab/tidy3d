import pydantic

""" Defines various validation functions that get used to ensure inputs are legit """

def ensure_greater_or_equal(field_name, value):
    """makes sure a field_name is >= value"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def is_greater_or_equal_to(val):
        assert (
            val >= value
        ), f"value of '{field_name}' must be greater than {value}, given {val}"
        return val

    return is_greater_or_equal_to


def ensure_less_than(field_name, value):
    """makes sure a field_name is less than value"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def is_less_than(field_val):
        assert (
            field_val < value
        ), f"value of '{field_name}' must be less than {value}, given {field_val}"
        return field_val

    return is_less_than


def assert_plane(field_name="geometry"):
    """makes sure a field's `size` attribute has exactly 1 zero"""

    @pydantic.validator(field_name, allow_reuse=True, always=True)
    def is_plane(cls, v):
        assert (
            v.size.count(0.0) == 1
        ), "mode objects only works with plane geometries with one size element of 0.0"
        return v

    return is_plane


def check_bounds():
    """makes sure the model's `bounds` field is Not none and is ordered correctly"""

    @pydantic.validator("bounds", allow_reuse=True)
    def valid_bounds(val):
        assert val is not None, "bounds must be set, are None"
        coord_min, coord_max = val
        for val_min, val_max in zip(coord_min, coord_max):
            assert val_min <= val_max, "min bound is smaller than max bound"
        return val

    return valid_bounds

def check_simulation_bounds():
    @pydantic.root_validator()
    def all_in_bounds(cls, values):
        sim_bounds = values.get("geometry").bounds
        sim_bmin, sim_bmax = sim_bounds

        check_objects = ("structures", "sources", "monitors")
        for obj_name in check_objects:

            # get all objects of name and continue if there are none
            objs = values.get(obj_name)
            if objs is None:
                continue

            # get bounds of each object
            for name, obj in objs.items():
                obj_bounds = obj.geometry.bounds
                obj_bmin, obj_bmax = obj_bounds

                # assert all of the object's max coordinates are greater than the simulation's min coordinate
                assert all(o >= s for (o, s) in zip(obj_bmax, sim_bmin)), f"{obj_name[:-1]} object '{name}' is outside of simulation bounds (on minus side)"

                # assert all of the object's min coordinates are less than than the simulation's max coordinate
                assert all(o <= s for (o, s) in zip(obj_bmin, sim_bmax)), f"{obj_name[:-1]} object '{name}' is outside of simulation bounds (on plus side)"

        return values
    return all_in_bounds