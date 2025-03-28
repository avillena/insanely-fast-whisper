"""
Utilidades y decoradores para el sistema de transcripción.
"""
import time
import logging
import functools
import importlib
from typing import Dict, Any, Callable
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Configurar consola
console = Console()

# Configuración de logging con Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            enable_link_path=False
        )
    ]
)
logger = logging.getLogger("transcriber")

# Utilidad para importación dinámica
def import_module(module_name: str) -> Any:
    """Importa un módulo dinámicamente y registra en el log."""
    logger.debug(f"Importando: {module_name}")
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Error importando {module_name}: {e}")
        raise

# Decorador para importar módulos justo antes de ejecutar la función
def with_imports(*module_names: str) -> Callable:
    """
    Decorador que importa módulos justo antes de ejecutar la función.
    
    Args:
        *module_names: Nombres de los módulos a importar
    
    Returns:
        Función decorada que tendrá los módulos importados disponibles
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dynamic_imports: Dict[str, Any] = {}
            for name in module_names:
                module_alias = name.split(".")[-1]
                dynamic_imports[module_alias] = import_module(name)
            kwargs["dynamic_imports"] = dynamic_imports
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Decorador para medir tiempo de ejecución
def log_time(func: Callable) -> Callable:
    """Decorador que mide el tiempo de ejecución de una función."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        logger.info(f"{func.__name__} - Tiempo: {end_time - start_time:.2f}s")
        return result
    return wrapper

# Función auxiliar para mostrar barra de progreso
def with_progress_bar(description: str, func: Callable) -> Any:
    """
    Ejecuta una función mostrando una barra de progreso.
    
    Args:
        description: Descripción para la barra de progreso
        func: Función a ejecutar
        
    Returns:
        Resultado de la función
    """
    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"[yellow]{description}", total=None)
        return func()