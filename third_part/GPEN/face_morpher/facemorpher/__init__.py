"""
Face Morpher module init code
"""
from .morpher import morpher, list_imgpaths
from .averager import averager

__all__ = ['list_imgpaths',
           'morpher',
           'averager']
