# Patch for adding preview functionality to upload section
# Replace the customization section in app_final.py around line 580-630

UPLOAD_PREVIEW_CODE = '''
            # Initialize upload preview state
            if 'upload_preview_figure' not in st.session_state:
                st.session_state.upload_preview_figure = None
            if 'upload_preview_config' not in st.session_state:
                st.session_state.upload_preview_config = None
            
            # Customization with Preview
            st.markdown("### 🎨 Chart Customization")
            custom_prompt = st.text_area(
                "Describe changes (preview before applying)",
                placeholder="e.g., Change title to 'Recent Inflation Trends', use blue colors, render averages for GDP and inflation",
                key="custom_upload"
            )
            
            col1_up, col2_up = st.columns(2)
            
            with col1_up:
                if st.button("🔍 Preview Changes", key="preview_upload"):
                    if custom_prompt:
                        with st.spinner("Generating preview..."):
                            try:
                                chart = st.session_state.current_upload_chart
                                chart_data = chart.get('data')
                                
                                updated_config = llm_handler.interpret_edit_request(
                                    custom_prompt,
                                    chart['config'],
                                    llm_provider,
                                    chart_data
                                )
                                
                                fig = chart_generator.create_chart(chart_data, updated_config)
                                
                                # Handle adding averages
                                if chart_data is not None and ('avg' in custom_prompt.lower() or 'average' in custom_prompt.lower()):
                                    numeric_cols = chart_data.select_dtypes(include='number').columns.tolist()
                                    for col in numeric_cols:
                                        if col in ['gdp_growth', 'inflation'] or any(term in col.lower() for term in ['gdp', 'inflation']):
                                            avg_val = chart_data[col].mean()
                                            fig.add_hline(
                                                y=avg_val,
                                                line_dash="dash",
                                                line_color="red" if 'inflation' in col.lower() else "blue",
                                                annotation_text=f"{col} avg: {avg_val:.2f}",
                                                annotation_position="top right"
                                            )
                                
                                st.session_state.upload_preview_figure = fig
                                st.session_state.upload_preview_config = updated_config
                                
                                st.success("✅ Preview generated! Review changes below.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Preview failed: {str(e)[:100]}...")
            
            with col2_up:
                if st.button("💾 Apply & Save Changes", key="apply_upload", disabled=st.session_state.upload_preview_figure is None):
                    if st.session_state.upload_preview_figure is not None:
                        with st.spinner("Saving changes..."):
                            try:
                                chart = st.session_state.current_upload_chart
                                
                                db.update_chart(
                                    chart['id'],
                                    chart_config=st.session_state.upload_preview_config,
                                    figure_json=st.session_state.upload_preview_figure.to_json()
                                )
                                
                                st.session_state.current_upload_chart['figure'] = st.session_state.upload_preview_figure
                                st.session_state.current_upload_chart['config'] = st.session_state.upload_preview_config
                                
                                st.session_state.upload_preview_figure = None
                                st.session_state.upload_preview_config = None
                                
                                st.success("✅ Changes applied and saved!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Save failed: {str(e)[:100]}...")
            
            # Show upload preview
            if st.session_state.upload_preview_figure is not None:
                st.markdown("#### 🔍 Preview")
                st.info("👆 Review the changes above. Click 'Apply & Save Changes' to make them permanent.")
                st.plotly_chart(st.session_state.upload_preview_figure, use_container_width=True, key="upload_preview_chart")
                
                if st.button("🔄 Reset Preview", key="reset_upload_preview"):
                    st.session_state.upload_preview_figure = None
                    st.session_state.upload_preview_config = None
                    st.rerun()
            
            st.markdown("---")
'''

# Instructions:
# 1. Find the upload section customization code (around line 580-630)
# 2. Replace the existing customization section with the code above
# 3. This adds preview functionality with separate state variables for upload section