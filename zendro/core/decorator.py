def node(
    func: Optional[Callable] = None,
    inputs: Optional[List[str]] = None,
    outputs: Optional[Union[str, List[str]]] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Union[Node, Callable]:
    """Decorator to create a node from a function."""
    
    # Handle case when decorator is used without parentheses
    if func is not None:
        return Node(
            func=func,
            inputs=inputs or [],
            outputs=[outputs] if isinstance(outputs, str) else (outputs or []),
            name=name or func.__name__,
            tags=tags or []
        )
    
    # Handle case when decorator is used with parentheses
    def decorator(function):
        return Node(
            func=function,
            inputs=inputs or [],
            outputs=[outputs] if isinstance(outputs, str) else (outputs or []),
            name=name or function.__name__,
            tags=tags or []
        )
    
    return decorator