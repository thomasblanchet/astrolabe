"""Register each hook of the plugin with mypy, by the full name of what it handles.

mypy has several kinds of plugin hook, and it asks for each kind in a
separate method of the plugin. This package has one class for each
kind: `TypeAnalyzeHooks`, `MethodHooks`, `FunctionHooks`,
`AttributeHooks` and `DynamicClassHooks`. Each class builds its hooks
and registers them in a `HookDispatcher`. The plugin builds these
classes once, and asks each one for the hook that matches a call.
"""
