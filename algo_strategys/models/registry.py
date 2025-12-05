"""
Unified Registry System for all components
所有模型組件都可以通過 registry 註冊和獲取
"""

from typing import Dict, Any, Type, Callable
import inspect


class Registry:
    """通用的註冊系統"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._config_registry: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str = None, config: Dict[str, Any] = None):
        """註冊裝飾器"""
        def decorator(cls):
            register_name = name or cls.__name__
            
            if register_name in self._registry:
                print(f"Warning: {register_name} already registered in {self.name}, overwriting")
            
            self._registry[register_name] = cls
            if config:
                self._config_registry[register_name] = config
            
            return cls
        return decorator
    
    def get(self, name: str, **kwargs) -> Any:
        """獲取註冊的組件"""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"{name} not found in {self.name} registry. "
                f"Available: {available}"
            )
        
        cls = self._registry[name]
        
        # 如果有默認配置，合併
        if name in self._config_registry:
            default_config = self._config_registry[name].copy()
            default_config.update(kwargs)
            kwargs = default_config
        
        return cls(**kwargs)
    
    def list(self) -> list:
        """列出所有註冊的組件"""
        return list(self._registry.keys())
    
    def has(self, name: str) -> bool:
        """檢查是否已註冊"""
        return name in self._registry

# 創建各種 registry
BACKBONE_REGISTRY = Registry("Backbone")                 # UNet backbone networks
TIME_EMBEDDING_REGISTRY = Registry("TimeEmbedding")      # Conditioning embedding networks (Modality: Time-series)
TEXT_EMBEDDING_REGISTRY = Registry("TextEmbedding")      # Conditioning embedding networks (Modality: Text)
POS_EMBEDDING_REGISTRY = Registry("Positional")          # Positional embedding
BLOCK_REGISTRY = Registry("Block")                       # Modules
ATTENTION_REGISTRY = Registry("Attention")               # Attention mechanism
SCHEDULER_REGISTRY = Registry("Scheduler")               # Diffusion scheduler

# Register functions
def register_backbone(name: str = None, **config):
    return BACKBONE_REGISTRY.register(name, config)
  
def register_time_embedding(name: str = None, **config):
    return TIME_EMBEDDING_REGISTRY.register(name, config)

def register_text_embedding(name: str = None, **config):
    return TEXT_EMBEDDING_REGISTRY.register(name, config)

def register_pos_embedding(name: str = None, **config):
    return POS_EMBEDDING_REGISTRY.register(name, config)

def register_block(name: str = None, **config):
    return BLOCK_REGISTRY.register(name, config)

def register_attention(name: str = None, **config):
    return ATTENTION_REGISTRY.register(name, config)

def register_scheduler(name: str = None, **config):
    return SCHEDULER_REGISTRY.register(name, config)

# Get model
def get_model(name: str, **kwargs):
    return MODEL_REGISTRY.get(name, **kwargs)

def get_time_embedding(name: str, **kwargs):
    return TIME_EMBEDDING_REGISTRY.get(name, **kwargs)

def get_text_embedding(name: str, **kwargs):
    return TEXT_EMBEDDING_REGISTRY.get(name, **kwargs)

def get_pos_embedding(name: str, **kwargs):
    return POS_EMBEDDING_REGISTRY.get(name, **kwargs)

def get_block(name: str, **kwargs):
    return BLOCK_REGISTRY.get(name, **kwargs)

def get_attention(name: str, **kwargs):
    return ATTENTION_REGISTRY.get(name, **kwargs)

def get_scheduler(name: str, **kwargs):
    return SCHEDULER_REGISTRY.get(name, **kwargs)

# 列出所有註冊的組件
def list_all_components():
    """列出所有可用的組件"""
    return {
        'models': MODEL_REGISTRY.list(),
        'time_embeddings': TIME_EMBEDDING_REGISTRY.list(),
        'text_embeddings': TEXT_EMBEDDING_REGISTRY.list(),
        'pos_embeddings': POS_EMBEDDING_REGISTRY.list(),
        'blocks': BLOCK_REGISTRY.list(),
        'attentions': ATTENTION_REGISTRY.list(),
        'schedulers': SCHEDULER_REGISTRY.list(),
    }
