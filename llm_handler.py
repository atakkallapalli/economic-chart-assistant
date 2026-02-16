import os
import json
from typing import Dict, List, Any, Optional
import boto3
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    """Handles LLM interactions for chart generation and summarization"""
    
    def __init__(self):
        self.bedrock_client = None
        self.anthropic_client = None
        self.openai_client = None
        
    def _get_bedrock_client(self):
        if not self.bedrock_client:
            region = os.getenv('AWS_REGION', 'us-east-1')
            # Use boto3's default credential chain which checks:
            # 1. ~/.aws/credentials file
            # 2. Environment variables
            # 3. IAM role (if on EC2)
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=region
            )
        return self.bedrock_client
    
    def _get_anthropic_client(self):
        if not self.anthropic_client:
            self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        return self.anthropic_client
    
    def _get_openai_client(self):
        if not self.openai_client:
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        return self.openai_client
    
    def interpret_chart_request(
        self, 
        user_prompt: str, 
        columns: List[str],
        data_sample: Dict,
        provider: str = "AWS Bedrock",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Interpret user's natural language request and generate chart configuration"""
        
        metadata_context = ""
        if metadata:
            metadata_context = f"\n\nMetadata/Stylesheet provided:\n{json.dumps(metadata, indent=2)}\n\nUse this metadata to guide chart styling and configuration."
        
        system_prompt = f"""You are an expert data visualization assistant for economic research.
Given a user's request and available data columns, generate a JSON configuration for creating a chart.

Available columns: {', '.join(columns)}
Data sample: {json.dumps(data_sample, indent=2)}{metadata_context}

Return ONLY a valid JSON object with this structure:
{{
    "chart_type": "line|bar|scatter|area|pie|heatmap",
    "x_column": "column_name",
    "y_columns": ["column1", "column2"],
    "title": "Chart Title",
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label",
    "colors": ["#color1", "#color2"],
    "show_legend": true|false,
    "annotations": []
}}"""

        user_message = f"Create a chart configuration for: {user_prompt}"
        
        response = self._call_llm(system_prompt, user_message, provider)
        
        try:
            # Extract JSON from response
            config = json.loads(response)
            return config
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not parse JSON from LLM response: {response}")
    
    def build_chart_interactively(
        self,
        user_prompt: str,
        data: Any,
        provider: str = "AWS Bedrock",
        metadata: Optional[Dict] = None,
        reference_image: Optional[str] = None
    ) -> Dict:
        """Build chart configuration interactively from prompt, data, metadata, and optional reference image"""
        
        # Extract data info
        columns = data.columns.tolist() if hasattr(data, 'columns') else []
        
        # Detect data format (wide vs long)
        data_format = "wide"
        component_names = []
        if 'variable' in columns and 'value' in columns:
            data_format = "long"
            unique_vars = data['variable'].unique().tolist() if 'variable' in data.columns else []
            component_names = unique_vars[:10]  # Limit to first 10
            data_context = f"Long format: 'variable' column contains component names: {', '.join(component_names)}"
        else:
            data_context = f"Wide format data"
        
        # Add metadata component info if available
        if metadata and 'data_configuration' in metadata and 'components' in metadata['data_configuration']:
            meta_components = [c.get('name', '') for c in metadata['data_configuration']['components']]
            data_context += f"\nMetadata components: {', '.join(meta_components)}"
        
        data_sample = data.head(5).to_dict('records') if hasattr(data, 'head') else {}
        
        metadata_context = ""
        if metadata:
            metadata_context = f"\n\nMetadata/Stylesheet:\n{json.dumps(metadata, indent=2)}"
        
        image_context = ""
        if reference_image:
            image_context = "\n\nA reference chart image has been provided showing the expected visual style and format."
        
        system_prompt = f"""You are an AI chart builder. Build a complete chart configuration from user prompt and data.

Data columns: {', '.join(columns)}
Data format: {data_context}
Data sample (first 5 rows): {json.dumps(data_sample, indent=2)}{metadata_context}{image_context}

IMPORTANT:
- The 'variable' column contains COMPONENT NAMES (e.g., "Housing Services", "Other Services", "Transportation Services")
- Each component name represents a different data series that will become a separate column after pivoting
- For PCE decomposition charts: use stacked_area chart type to show component contributions over time
- After data is pivoted, x_column='date' and y_columns should list the component names
- Always ensure x_column and y_columns exist in the data after transformation

Return ONLY valid JSON:
{{
    "chart_type": "stacked_area|line|bar|scatter|area",
    "x_column": "date",
    "y_columns": ["component1", "component2", ...],
    "title": "Chart Title",
    "x_label": "X Label",
    "y_label": "Y Label",
    "colors": ["#2E86AB", "#A23B72"],
    "show_legend": true
}}"""
        
        user_message = f"Build chart for: {user_prompt}"
        
        # If reference image provided, use vision-capable call
        if reference_image:
            response = self._call_llm_with_image(system_prompt, user_message, reference_image, provider)
        else:
            response = self._call_llm(system_prompt, user_message, provider)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                config = json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                config = json.loads(json_str)
            else:
                config = json.loads(response)
            
            # Validate and fix config
            if 'x_column' not in config or config['x_column'] not in columns:
                # Auto-detect x column
                if 'date' in columns:
                    config['x_column'] = 'date'
                elif len(columns) > 0:
                    config['x_column'] = columns[0]
            
            if 'y_columns' not in config or not config['y_columns']:
                # Auto-detect y columns
                if 'value' in columns:
                    config['y_columns'] = ['value']
                else:
                    numeric_cols = [c for c in columns if c != config.get('x_column')]
                    config['y_columns'] = numeric_cols[:1] if numeric_cols else [columns[1] if len(columns) > 1 else columns[0]]
            
            return config
        except Exception as e:
            # Fallback: create basic config
            return {
                'chart_type': 'line',
                'x_column': 'date' if 'date' in columns else columns[0],
                'y_columns': ['value'] if 'value' in columns else [columns[1] if len(columns) > 1 else columns[0]],
                'title': user_prompt[:50],
                'x_label': 'Date',
                'y_label': 'Value',
                'show_legend': True
            }
    
    def interpret_edit_request(
        self,
        edit_prompt: str,
        current_config: Dict,
        provider: str = "AWS Bedrock",
        chart_data: Optional[Any] = None
    ) -> Dict:
        """Interpret edit request and update chart configuration with data context"""
        
        try:
            data_context = ""
            if chart_data is not None:
                try:
                    if hasattr(chart_data, 'columns'):
                        data_context += f"\nAvailable data columns: {', '.join(chart_data.columns.tolist())}"
                    if hasattr(chart_data, 'dtypes'):
                        numeric_cols = chart_data.select_dtypes(include='number').columns.tolist()
                        data_context += f"\nNumeric columns: {', '.join(numeric_cols)}"
                except Exception:
                    pass
            
            system_prompt = f"""You are an expert at modifying chart configurations based on user requests.
Given the current configuration and an edit request, return the updated configuration.
{data_context}

Return ONLY a valid JSON object with the complete updated configuration."""
            
            user_message = f"""Current configuration:
{json.dumps(current_config, indent=2)}

Edit request: {edit_prompt}

Return the complete updated configuration as JSON."""
            
            response = self._call_llm(system_prompt, user_message, provider)
            
            if not response:
                raise ValueError("LLM returned empty response")
            
            try:
                config = json.loads(response)
                return config
            except json.JSONDecodeError:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                    return json.loads(json_str)
                else:
                    raise ValueError(f"Could not parse JSON from LLM response: {response[:200]}")
        except Exception as e:
            raise Exception(f"Edit request failed: {str(e)}")
    
    def generate_summary(
        self,
        charts: List[Dict],
        data: Any,
        summary_type: str,
        length: str,
        provider: str = "AWS Bedrock",
        custom_prompt: Optional[str] = None,
        chart_data_list: Optional[List[Any]] = None
    ) -> str:
        """Generate summary of selected charts with enhanced data context"""
        
        chart_descriptions = []
        for idx, chart in enumerate(charts):
            description = f"""
Chart {idx + 1}:
- Original request: {chart['prompt']}
- Configuration: {json.dumps(chart['config'], indent=2)}
"""
            
            # Add data context if available
            if chart_data_list and idx < len(chart_data_list) and chart_data_list[idx] is not None:
                try:
                    chart_data = chart_data_list[idx]
                    if hasattr(chart_data, 'describe'):
                        data_summary = chart_data.describe().to_string()
                        description += f"\n- Data Summary:\n{data_summary[:300]}..."
                except Exception:
                    pass
            
            chart_descriptions.append(description)
        
        length_instructions = {
            "Brief": "Keep the summary concise (2-3 paragraphs)",
            "Medium": "Provide a moderate length summary (4-6 paragraphs)",
            "Detailed": "Provide a comprehensive detailed analysis (8-10 paragraphs)"
        }
        
        type_instructions = {
            "Executive Summary": "Focus on high-level insights and key takeaways for executives",
            "Detailed Analysis": "Provide in-depth analysis of trends, patterns, and implications",
            "Key Insights": "List the most important insights and findings",
            "Custom": custom_prompt or "Provide a comprehensive summary"
        }
        
        system_prompt = f"""You are an expert economic analyst. Analyze the provided charts and data to generate insights.

{type_instructions[summary_type]}
{length_instructions[length]}

Focus on:
- Key trends and patterns
- Economic implications
- Data-driven insights
- Actionable recommendations"""

        user_message = f"""Analyze these charts and provide a summary:

{chr(10).join(chart_descriptions)}

Data summary:
- Rows: {len(data)}
- Columns: {', '.join(data.columns.tolist())}
- Date range: {data.iloc[0, 0] if len(data) > 0 else 'N/A'} to {data.iloc[-1, 0] if len(data) > 0 else 'N/A'}

Generate the summary now."""

        response = self._call_llm(system_prompt, user_message, provider)
        return response
    
    def answer_chart_question(
        self,
        question: str,
        charts: List[Dict],
        context_type: str = "General Analysis",
        detail_level: str = "Brief",
        provider: str = "AWS Bedrock",
        chart_data: Optional[Any] = None
    ) -> str:
        """Answer questions about specific charts using LLM analysis"""
        
        # Prepare chart information with data context
        chart_info = []
        for idx, chart in enumerate(charts):
            info = f"""
Chart {idx + 1}: {chart.get('chart_name', f'Chart {idx+1}')}
- Type: {chart.get('chart_config', {}).get('chart_type', 'Unknown')}
- Original Request: {chart.get('user_prompt', 'N/A')}
- Configuration: {json.dumps(chart.get('chart_config', {}), indent=2)}
- Created: {chart.get('created_at', 'N/A')}
"""
            
            # Add data summary if available
            if chart_data is not None:
                try:
                    if hasattr(chart_data, 'describe'):
                        data_summary = chart_data.describe().to_string()
                        info += f"\n- Data Summary:\n{data_summary[:500]}..."
                    if hasattr(chart_data, 'columns'):
                        info += f"\n- Columns: {', '.join(chart_data.columns.tolist())}"
                    if hasattr(chart_data, 'shape'):
                        info += f"\n- Data Shape: {chart_data.shape[0]} rows, {chart_data.shape[1]} columns"
                except Exception:
                    pass
            
            chart_info.append(info)
        
        # Context-specific instructions
        context_instructions = {
            "General Analysis": "Provide a comprehensive overview and general insights",
            "Trend Analysis": "Focus on identifying and explaining trends, patterns, and temporal changes",
            "Comparative Analysis": "Compare and contrast different data series, time periods, or categories",
            "Statistical Insights": "Provide statistical analysis, correlations, and quantitative insights",
            "Economic Implications": "Focus on economic meaning, policy implications, and business impact"
        }
        
        # Detail level instructions
        detail_instructions = {
            "Brief": "Provide a concise, focused answer (2-3 paragraphs)",
            "Detailed": "Provide a comprehensive analysis with supporting details (4-6 paragraphs)",
            "Technical": "Include technical details, methodology, and statistical considerations"
        }
        
        system_prompt = f"""You are an expert data analyst and economic researcher. 
Analyze the provided charts and answer the user's question with precision and insight.

{context_instructions[context_type]}
{detail_instructions[detail_level]}

Guidelines:
- Base your analysis on the chart configurations and data patterns
- Use the provided data summaries and statistics when available
- Provide specific, actionable insights
- Reference specific charts when making points
- Use clear, professional language
- Support conclusions with evidence from the charts and data"""

        user_message = f"""Question: {question}

Chart Information:
{chr(10).join(chart_info)}

Please analyze these charts and provide a comprehensive answer to the question."""

        response = self._call_llm(system_prompt, user_message, provider)
        return response
    
    def _call_llm(self, system_prompt: str, user_message: str, provider: str) -> str:
        """Call the appropriate LLM based on provider"""
        
        if provider == "AWS Bedrock":
            return self._call_bedrock(system_prompt, user_message)
        elif provider == "Anthropic":
            return self._call_anthropic(system_prompt, user_message)
        elif provider == "OpenAI":
            return self._call_openai(system_prompt, user_message)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_bedrock(self, system_prompt: str, user_message: str) -> str:
        """Call AWS Bedrock Claude"""
        try:
            client = self._get_bedrock_client()
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
            
            response = client.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            raise Exception(f"Bedrock API error: {str(e)}")
    
    def _call_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Call Anthropic API directly"""
        client = self._get_anthropic_client()
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )
        
        return message.content[0].text
    
    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI API"""
        client = self._get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=4000
        )
        
        return response.choices[0].message.content
    
    def _call_llm_with_image(self, system_prompt: str, user_message: str, image_base64: str, provider: str) -> str:
        """Call LLM with image support for reference chart"""
        
        if provider == "AWS Bedrock":
            return self._call_bedrock_with_image(system_prompt, user_message, image_base64)
        elif provider == "Anthropic":
            return self._call_anthropic_with_image(system_prompt, user_message, image_base64)
        elif provider == "OpenAI":
            return self._call_openai_with_image(system_prompt, user_message, image_base64)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_bedrock_with_image(self, system_prompt: str, user_message: str, image_base64: str) -> str:
        """Call AWS Bedrock Claude with image"""
        try:
            client = self._get_bedrock_client()
            
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": user_message
                            }
                        ]
                    }
                ]
            }
            
            response = client.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            raise Exception(f"Bedrock API error: {str(e)}")
    
    def _call_anthropic_with_image(self, system_prompt: str, user_message: str, image_base64: str) -> str:
        """Call Anthropic API with image"""
        client = self._get_anthropic_client()
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": user_message
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    def _call_openai_with_image(self, system_prompt: str, user_message: str, image_base64: str) -> str:
        """Call OpenAI API with image"""
        client = self._get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_message
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        
        return response.choices[0].message.content
