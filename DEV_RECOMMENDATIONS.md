# Step Workflow Development Recommendations

## High Level Concepts

1. Steps are for serialization.

    Unlike other workflow management systems where you can treat individual points in
    the workflow as tiny, sometimes even anonymous lambda functions, treat steps in
    `datastep` workflows as "points at which you would like to serialize something to
    disk". This is what makes it easy for the system to work well with large data backed
    projects, the code itself, and quilt (the data storage system) to work well
    together.

    This means, if you want to run an entire flow inside your step go ahead, the end
    result should be a manifest of files that act as a nice packaged unit of work and
    data.

2. Steps should run functionality from other Python packages.

    Not always true, but in general try to keep the workflow simply as just a "method to
    manage large processing trees" rather than also keeping the code to do the
    processing itself. This is to make it easier for development to continue with many
    people working on the project and, because there is an assumption that testing
    occurs on the imported packages, makes running the test suite on the workflow
    itself much shorter.

    The obvious exception to this concept is where the underlying package actually
    utilizes `datastep` for the processing as well (Mitotic Classifier, NMA, etc.).

## FAQ

1. "I added a task to the `direct_upstream_tasks` parameter in my step but it isn't
being ran during `all`. What's wrong?"

    The `direct_upstream_tasks` parameter is used for _pulling_ data down from Quilt so
    that you can experiment on the step itself. To add the step you created to the
    pipeline add to the `bin/all.py` script.

2. "I added my step to `bin/all.py` but it isn't running in the correct order or with
the needed data. How to I tell it which data it needs?"

    Treat steps like normal Python functions. They require inputs and return outputs.
    To tell the step that it requires upstream data to actually start it's processing,
    simply add the return value of upstream data as an input to your own step.

3. "Are step workflows linear or can they have branches?"

    `datastep` workflows have full DAG / workflow functionality. You can create branches
    by simply using the output of a prior step in your own step.


## Other Tips

1. When creating a new steps that has many words in the step name try putting
underscores (`_`) in between the words when creating it. This will result in
standardized file name and step names (ex: `make_new_step this_is_a_step` results
in file: `cvapipe/steps/this_is_a_step.py` and step name: `ThisIsAStep`).
