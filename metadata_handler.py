import json
import yaml
from typing import Dict, Any, Optional

class MetadataHandler:
    """Handle metadata/stylesheet for chart generation"""
    
    @staticmethod
    def parse_metadata(file_content: str, file_type: str) -> Dict[str, Any]:
        """Parse metadata from JSON or YAML"""
        try:
            if file_type in ['json']:
                metadata = json.loads(file_content)
            elif file_type in ['yaml', 'yml']:
                metadata = yaml.safe_load(file_content)
            else:
                raise ValueError(f"Unsupported metadata format: {file_type}")
            
            # Validate and normalize structure
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            
            return metadata
        except Exception as e:
            raise ValueError(f"Failed to parse metadata: {str(e)}")
    
    @staticmethod
    def merge_with_config(base_config: Dict, metadata: Dict) -> Dict:
        """Merge metadata with base chart config"""
        merged = base_config.copy()
        
        # Chart styling
        if 'style' in metadata:
            style = metadata['style']
            if 'colors' in style and style['colors']:
                merged['colors'] = style['colors']
            if 'palette' in style:
                merged['color_palette'] = style['palette']
            if 'title' in style:
                merged['title'] = style['title']
            if 'x_label' in style:
                merged['x_label'] = style['x_label']
            if 'y_label' in style:
                merged['y_label'] = style['y_label']
            if 'show_legend' in style:
                merged['show_legend'] = style['show_legend']
        
        # Chart type and columns
        if 'chart' in metadata:
            chart = metadata['chart']
            if 'type' in chart:
                merged['chart_type'] = chart['type']
            if 'x_column' in chart:
                merged['x_column'] = chart['x_column']
            if 'y_columns' in chart:
                merged['y_columns'] = chart['y_columns']
        
        # Annotations
        if 'annotations' in metadata:
            merged['annotations'] = metadata['annotations']
        
        # Custom properties
        if 'custom' in metadata:
            merged['custom'] = metadata['custom']
        
        return merged
