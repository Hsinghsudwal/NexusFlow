class Step:
    def __init__(self, name, function, inputs=None, outputs=None):
        self.name = name
        self.function = function
        self.inputs = inputs or []
        self.outputs = outputs or []

    def run(self, context):
        return self.function(context)

class Pipeline:
    def __init__(self, name, steps):
        self.name = name
        self.steps = steps

    def run(self, context):
        for step in self.steps:
            context = step.run(context)
        return context
