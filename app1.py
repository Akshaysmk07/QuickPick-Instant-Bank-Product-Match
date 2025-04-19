import streamlit as st
import recommendation_system as r
import pandas as pd

# Page configuration with custom theme
st.set_page_config(
    page_title="QuickPick Product Recommender",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #EC0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .product-section {
        padding: 1rem;
        background-color: #fff;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .recommendation-item {
        padding: 0.8rem;
        background-color: #FFF2F2;
        border-left: 4px solid #EC0000;
        margin-bottom: 0.5rem;
        border-radius: 0 5px 5px 0;
    }
    .highlight {
        background-color: #EC0000;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #666;
    }
    .logo {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Caching data loading
@st.cache_data
def get_data():
    df_train1505 = r.load_csv()
    return df_train1505

# App Header with bank icon (using emoji instead of image)
st.markdown('<div class="logo"><h1>🏦</h1></div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">QuickPick: Instant Bank Product Match</h1>', unsafe_allow_html=True)

# Description
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem;">
            Speed up your decision-making with instant, tailored banking product suggestions that fit your unique financial profile.
        </p>
    </div>
""", unsafe_allow_html=True)

# Introduction in a card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('''
    <h3>Welcome to QuickPick</h3>
    
    This hybrid recommendation system analyzes your current product portfolio to suggest additional banking products 
    that might benefit you. Our advanced system uses three complementary engines:
    
    🔍 **User Similarity Engine**: Finds patterns between customers with similar product preferences
    
    🤖 **Machine Learning Classifier**: Makes intelligent predictions based on existing customer data
    
    📊 **Popularity Analysis**: Identifies widely adopted products across our customer base
    
    These engines work together to provide you with personalized recommendations tailored to your needs.
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Instructions
# Instructions
st.markdown('<h3 class="subheader" style="color: white;">Select the products you currently own:</h3>', unsafe_allow_html=True)

# Product categories
product_list = [product for product in r.product_names.values()]
product_categories = {
    "Banking Products": product_list[:8],
    "Investment Products": product_list[8:16],
    "Insurance & Services": product_list[16:]
}

# Initialize session state for selections if not already done
if 'product_selections' not in st.session_state:
    st.session_state.product_selections = [0] * len(product_list)

# Create tabs for product categories
tabs = st.tabs(list(product_categories.keys()))

# Display products in tabs
for i, (category, products) in enumerate(product_categories.items()):
    with tabs[i]:
        st.markdown('<div class="product-section">', unsafe_allow_html=True)
        cols = st.columns(2)
        
        for j, product in enumerate(products):
            col_idx = j % 2
            product_idx = list(product_categories.keys()).index(category) * 8 + j
            
            with cols[col_idx]:
                # Update session state when selection changes
                current_value = st.session_state.product_selections[product_idx]
                new_value = 1 if st.toggle(f"{product}", value=(current_value == 1)) else 0
                st.session_state.product_selections[product_idx] = new_value

        st.markdown('</div>', unsafe_allow_html=True)

# Summary of selected products
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3>Your Selected Products</h3>', unsafe_allow_html=True)

selected_count = sum(st.session_state.product_selections)
if selected_count > 0:
    selected_products = [product_list[i] for i in range(len(product_list)) if st.session_state.product_selections[i] == 1]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        for product in selected_products:
            st.markdown(f"✅ {product}")
    with col2:
        st.metric("Products Selected", f"{selected_count}/{len(product_list)}")
else:
    st.info("You haven't selected any products yet. Please select the products you currently own.")
st.markdown('</div>', unsafe_allow_html=True)

# Get recommendations button
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        click = st.button('Generate Personalized Recommendations', type="primary", use_container_width=True)

# Process and display recommendations
if click:
    with st.spinner('Analyzing your profile and calculating personalized recommendations...'):
        # Check if user already has all products
        if sum(st.session_state.product_selections) == len(product_list):
            st.success('🎉 Congratulations! You already have all available banking products.')
        else:
            # Get recommendations
            df_train1505 = get_data()
            df_train1505 = r.add_user_input(st.session_state.product_selections, df_train1505)
            df_ui = r.df_useritem(df_train1505)
            cos_sim = r.cos_sim(df_ui)
            ui = r.useritem(0, df_ui, sim_matrix=cos_sim)
            df_mb = r.df_mb(df_train1505)

            hybrid_rec = r.hybrid(0, df_p=df_train1505, df_u=df_ui, sim_matrix=cos_sim, 
                                 df_m=df_mb, f1=0.5, f2=0.25, f3=0.25)

            recommendations = r.recommendation(0, df_mb, hybrid_rec)

            # Display recommendations in a nice format
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Your Personalized Recommendations</h3>', unsafe_allow_html=True)
            st.markdown('<p>Based on your current product portfolio, we recommend the following:</p>', unsafe_allow_html=True)
            
            for idx, product in enumerate(recommendations):
                st.markdown(f'<div class="recommendation-item">'
                            f'<span class="highlight">{idx + 1}</span> &nbsp; <b style="color: black;">{product}</b>'
                            f'</div>', unsafe_allow_html=True)
                
            st.info("These recommendations are based on customer behavior patterns and product synergies. Please consult with a banking advisor for personalized financial advice.")
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 QuickPick: Instant Bank Product Match | Powered by Streamlit</div>', 
           unsafe_allow_html=True)