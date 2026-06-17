# =============================================================================
# Semester Project: Intelligent Data-Driven Web Chatbot for Motorcycle Spare Parts Shop
# Target Domain: Motorcycle Spare Parts & Accessories Business (Pakistan Market Rates)
# Framework: Streamlit Web UI (Migrated from CLI)
# Architecture: 9-Step Modular GUI Pipeline
# Author: BS Artificial Intelligence Student
# =============================================================================

# ==========================================
# STEP 1: LIBRARY IMPORTS & SETUP
# ==========================================
import streamlit as st
import pandas as pd
import os  
import math
import re
from collections import Counter

def get_whatsapp_link(phone_number):
    """Formats phone number into a direct WhatsApp chat link."""
    clean_number = re.sub(r'\D', '', phone_number)
    if clean_number.startswith('0'):
        clean_number = '92' + clean_number[1:]
    return f"https://wa.me/{clean_number}"

# Set page config at the absolute top of the file
st.set_page_config(
    page_title="Motorcycle Parts AI Assistant", 
    page_icon="⚙", 
    layout="wide"
)

# ==========================================
# STEP 2: THE 200 FAQ DATASET INITIALIZATION
# ==========================================
@st.cache_data
def initialize_faq_dataset():
    """
    Generates 200 high-quality, professional FAQ entries.
    5 Intents x 40 Parts = 200 distinct, owner-verified QA pairs.
    """
    parts_pool = [
        "Air Cleaner Filter", "Drum Rubber", "Disc Pad", "Piston Kit", "Mudguard Front", 
        "Cylinder Block", "Drive Chain", "Side Cover", "Indicator Relay", "Spark Plug", 
        "Brake Cable", "Rear Axle", "Headlight Assembly", "Oil Pump", "CDI Unit", 
        "Brake Lever", "Fuel Tank", "Footrest Rubber", "Brake Shoe Set", "Handle Bar", 
        "Handle Grip", "Mudguard Rear", "Clutch Plate Set", "Ignition Switch", "Gasket Kit", 
        "Sprocket Hub", "Chain Sprocket Kit", "Engine Oil 1L", "Pressure Plate", "Piston Ring Set", 
        "Front Shock Absorber", "Rear Shock Absorber", "Accelerator Cable", "Clutch Cable", 
        "Speedometer Cable", "Backlight Assembly", "Turn Signal Indicator", "Fuel Tank Cap", 
        "Side Mirror Set", "Main Center Stand"
    ]
    
    intents = [
        {
            "intent": "stock", 
            "q": "Do you have {} in stock?", 
            "a": "Yes, we currently maintain {} in our active inventory. Would you like me to check the current unit price for you?"
        },
        {
            "intent": "price", 
            "q": "What is the price of a {}?", 
            "a": "The current market rate for our {} is competitive. We source high-quality parts; please let me know if you need specific compatibility details."
        },
        {
            "intent": "availability", 
            "q": "Can you check the availability of a {}?", 
            "a": "I am checking our warehouse database for {}. We update our stock levels daily to ensure you get the latest parts."
        },
        {
            "intent": "compatibility", 
            "q": "Is the {} compatible with Honda CD70 or CG125?", 
            "a": "Most of our {} variants are precision-engineered. While some are model-specific, many of our parts offer cross-compatibility for standard 70cc and 125cc commuter motorcycles."
        },
        {
            "intent": "warranty", 
            "q": "Do you offer any warranty on your {}?", 
            "a": "All our mechanical components, including the {}, come with a performance guarantee. If you face any manufacturing defects, please reach out to our support team."
        }
    ]
    
    faq_list = []
    for intent in intents:
        for part in parts_pool:
            faq_list.append({
                "intent": intent["intent"],
                "part_keyword": part,
                "question": intent["q"].format(part),
                "answer": intent["a"].format(part)
            })
            
    return faq_list

# ==========================================
# STEP 3: ADVANCED TF-IDF VECTORIZATION ENGINE
# ==========================================
def clean_text(text):
    """Cleans strings by normalizing cases, stripping punctuation, and breaking into words."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

@st.cache_data
def compute_idf(faq_dataset):
    """Calculates Inverse Document Frequency (IDF) for all words across the 200 FAQs."""
    N = len(faq_dataset)
    usage_count = Counter()
    
    for faq in faq_dataset:
        unique_tokens = set(clean_text(faq["question"]))
        for token in unique_tokens:
            usage_count[token] += 1
            
    idf = {}
    for token, count in usage_count.items():
        idf[token] = math.log(N / (1 + count)) + 1.0
    return idf

def text_to_tfidf_vector(tokens, idf_matrix):
    """Converts token streams into explicit TF-IDF weighted vector dictionaries."""
    tf = Counter(tokens)
    tfidf_vector = {}
    for token, freq in tf.items():
        idf_weight = idf_matrix.get(token, 1.0) 
        tfidf_vector[token] = freq * idf_weight
    return tfidf_vector

def compute_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity metric between two token-frequency vector dictionaries."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    dot_product = sum(vec1[x] * vec2[x] for x in intersection)
    magnitude1 = sum(val ** 2 for val in vec1.values())
    magnitude2 = sum(val ** 2 for val in vec2.values())
    
    if not magnitude1 or not magnitude2:
        return 0.0
    
    return dot_product / (math.sqrt(magnitude1) * math.sqrt(magnitude2))

def find_best_faq_match(user_query, faq_dataset, idf_matrix, inventory_df):
    """Maps search inputs onto localized TF-IDF spaces and dynamically hooks live metrics into final responses."""
    user_tokens = clean_text(user_query)
    if not user_tokens:
        return "I'm sorry, I couldn't understand your question. Could you please rephrase it?"
        
    query_lower = user_query.lower()
    
    # INTERCEPT QUERY FOR GLOBAL OUT-OF-STOCK PRODUCT LISTING
    out_of_stock_keywords = ["out of stock", "short out", "depleted", "zero stock", "missing items", "which parts are out", "show items with 0 stock"]
    if any(keyword in query_lower for keyword in out_of_stock_keywords):
        shorted_items = inventory_df[inventory_df["Stock Quantity"] == 0]
        if shorted_items.empty:
            return "🎉 **Live Inventory Status:** Excellent news! All tracked spare parts are currently fully supplied. No items are completely shorted out today."
        
        response_text = f"🚨 **Live System Alert:** There are currently **{len(shorted_items)} parts** completely out of stock today. Here is the explicit product list:\n\n"
        for idx, row in shorted_items.iterrows():
            model_info = f" ({row['Model_Compatibility']})" if 'Model_Compatibility' in row else ""
            cat_info = f" Class: *{row['Category']}*" if 'Category' in row else ""
            response_text += f"* ❌ **{row['Item Name']}{model_info}** | {cat_info} — Base Rate: Rs. {int(row['Rate']):,}\n"
        response_text += "\n💡 *These variants require an immediate procurement restock order.*"
        return response_text

    user_vector = text_to_tfidf_vector(user_tokens, idf_matrix)
    
    best_score = -1.0
    matched_node = None
    
    for faq in faq_dataset:
        faq_tokens = clean_text(faq["question"])
        faq_vector = text_to_tfidf_vector(faq_tokens, idf_matrix)
        
        score = compute_cosine_similarity(user_vector, faq_vector)
        if score > best_score:
            best_score = score
            matched_node = faq
            
    if best_score < 0.15:
        return ("I couldn't find an exact matching question in our 200-FAQ database. "
                "Try asking directly about part availability, prices, or compatibility (e.g., 'Do you have Spark Plug in stock?').")
                
    intent_type = matched_node["intent"]
    part_keyword = matched_node["part_keyword"]
    base_reply = matched_node["answer"]
    
    keyword_tokens = set(clean_text(part_keyword))
    full_keyword_lower = part_keyword.lower()
    
    best_db_match = None
    highest_overlap = 0
    
    mentioned_model = None
    if "70cc" in query_lower or "70" in query_lower:
        mentioned_model = "70cc"
    elif "125cc" in query_lower or "125" in query_lower:
        mentioned_model = "125cc"
    elif "100cc" in query_lower or "100" in query_lower:
        mentioned_model = "100cc"
    
    for idx, row in inventory_df.iterrows():
        row_name_str = str(row['Item Name'])
        row_tokens = set(clean_text(row_name_str))
        row_model = str(row.get('Model_Compatibility', '')).lower()
        
        if mentioned_model and mentioned_model != row_model:
            continue
            
        common_tokens = keyword_tokens.intersection(row_tokens)
        overlap_count = len(common_tokens)
        
        if full_keyword_lower in row_name_str.lower():
            best_db_match = row
            break
        elif overlap_count > highest_overlap:
            highest_overlap = overlap_count
            best_db_match = row
            
    if best_db_match is None and mentioned_model is not None:
        for idx, row in inventory_df.iterrows():
            row_name_str = str(row['Item Name'])
            row_tokens = set(clean_text(row_name_str))
            common_tokens = keyword_tokens.intersection(row_tokens)
            overlap_count = len(common_tokens)
            
            if full_keyword_lower in row_name_str.lower():
                best_db_match = row
                break
            elif overlap_count > highest_overlap:
                highest_overlap = overlap_count
                best_db_match = row
                
    if best_db_match is not None:
        live_name = best_db_match['Item Name']
        if 'Model_Compatibility' in best_db_match:
            live_name = f"{live_name} ({best_db_match['Model_Compatibility']})"
            
        live_price = best_db_match['Rate']
        live_qty = best_db_match['Stock Quantity']
        
        if intent_type == "price":
            base_reply = f"The live system price for **{live_name}** is **Rs. {live_price:,}**. We offer premium OEM and verified aftermarket options."
        elif intent_type in ["stock", "availability"]:
            if live_qty > 0:
                base_reply = f"Yes, **{live_name}** is in stock! We currently have **{live_qty} units** ready for immediate dispatch at **Rs. {live_price:,} per unit**."
            else:
                base_reply = f"Currently, **{live_name}** is out of stock (**0 units remaining**). Please check our 'Short Out Alerts' tracking dashboard."
                
    return f"**Matched Question:** *\"{matched_node['question']}\"* \n\n💬 **AI Response:** {base_reply}"


# ==========================================
# STEP 4: INVENTORY MANAGEMENT SYSTEM SETUP
# ==========================================
@st.cache_data
def load_inventory_data():
    """Data Science Pipeline: Dynamic CSV Data Loading with Template Fallbacks."""
    csv_filename = "motorcycle_spare_parts_dataset_accurate.csv"
    
    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename)
            rename_map = {
                'Part_Name': 'Item Name',
                'Price_PKR': 'Rate',
                'Stock_Quantity': 'Stock Quantity'
            }
            df = df.rename(columns=rename_map)
            
            if 'Item Name' in df.columns and 'Rate' in df.columns and 'Stock Quantity' in df.columns:
                df['Rate'] = df['Rate'].round().astype(int)
                if 'Minimum Required Level' not in df.columns:
                    df['Minimum Required Level'] = 15
                return df[['Item Name', 'Category', 'Model_Compatibility', 'Rate', 'Stock Quantity', 'Minimum Required Level']]
        except Exception:
            pass 

    fallback_data = [
        {"Item Name": "Spark Plug", "Category": "Electrical", "Model_Compatibility": "70cc", "Rate": 350, "Stock Quantity": 45, "Minimum Required Level": 15},
        {"Item Name": "Air Cleaner Filter", "Category": "Rubber/Misc", "Model_Compatibility": "70cc", "Rate": 160, "Stock Quantity": 141, "Minimum Required Level": 15},
        {"Item Name": "Drum Rubber", "Category": "Rubber/Misc", "Model_Compatibility": "125cc", "Rate": 220, "Stock Quantity": 74, "Minimum Required Level": 15},
        {"Item Name": "Disc Pad", "Category": "Braking", "Model_Compatibility": "125cc", "Rate": 450, "Stock Quantity": 61, "Minimum Required Level": 15},
        {"Item Name": "Piston Kit", "Category": "Engine", "Model_Compatibility": "125cc", "Rate": 1850, "Stock Quantity": 61, "Minimum Required Level": 15},
        {"Item Name": "Mudguard Front", "Category": "Body/Frame", "Model_Compatibility": "100cc", "Rate": 750, "Stock Quantity": 247, "Minimum Required Level": 15},
        {"Item Name": "Cylinder Block", "Category": "Engine", "Model_Compatibility": "100cc", "Rate": 3800, "Stock Quantity": 294, "Minimum Required Level": 15},
        {"Item Name": "Drive Chain", "Category": "Drivetrain", "Model_Compatibility": "100cc", "Rate": 600, "Stock Quantity": 190, "Minimum Required Level": 15},
        {"Item Name": "Side Cover", "Category": "Body/Frame", "Model_Compatibility": "125cc", "Rate": 1350, "Stock Quantity": 371, "Minimum Required Level": 15},
        {"Item Name": "Indicator Relay", "Category": "Electrical", "Model_Compatibility": "70cc", "Rate": 90, "Stock Quantity": 0, "Minimum Required Level": 15},
        {"Item Name": "Brake Cable", "Category": "Braking", "Model_Compatibility": "125cc", "Rate": 180, "Stock Quantity": 0, "Minimum Required Level": 15}
    ]
    return pd.DataFrame(fallback_data)


# ==========================================
# STEP 5: CHECK AVAILABLE INVENTORY (FEATURE 1)
# ==========================================
def display_full_inventory(df):
    st.subheader("📋 Complete Inventory Catalog")
    st.write("Browse, filter, and audit our entire stock catalog below.")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        categories = ["All"] + list(df['Category'].unique()) if 'Category' in df.columns else ["All"]
        selected_cat = st.selectbox("📂 Segment Category Filter", categories)
    with col_f2:
        models = ["All"] + list(df['Model_Compatibility'].unique()) if 'Model_Compatibility' in df.columns else ["All"]
        selected_model = st.selectbox("🏍 Engine Model Suitability", models)
    search_query = st.text_input("🔍 Quick Search Item by Name", "").strip().lower()
    filtered_df = df.copy()
    if 'Category' in df.columns and selected_cat != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_cat]
    if 'Model_Compatibility' in df.columns and selected_model != "All":
        filtered_df = filtered_df[filtered_df['Model_Compatibility'] == selected_model]
    if search_query:
        filtered_df = filtered_df[filtered_df['Item Name'].str.lower().str.contains(search_query)]
    display_cols = ['Item Name']
    if 'Category' in df.columns: display_cols.append('Category')
    if 'Model_Compatibility' in df.columns: display_cols.append('Model_Compatibility')
    display_cols.extend(['Rate', 'Stock Quantity'])
    st.dataframe(
        filtered_df[display_cols], 
        column_config={
            "Item Name": "Spare Part Item Description",
            "Category": "Category Class",
            "Model_Compatibility": "Compatibility Model",
            "Rate": st.column_config.NumberColumn("Unit Price (Rs.)", format="Rs. %d"),
            "Stock Quantity": st.column_config.ProgressColumn("Stock Quantity Level", format="%d units", min_value=0, max_value=500)
        },
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# STEP 6: SHORT OUT / LOW STOCK ALERTS (FEATURE 2)
# ==========================================
def display_short_out_alerts(df):
    st.subheader("🚨 Stock Alerts & Reorder Notifications")
    
    # 0 stock wale items
    out_of_stock = df[df["Stock Quantity"] == 0]
    # Reorder level se kam stock wale items (excluding already 0)
    low_stock = df[(df["Stock Quantity"] > 0) & (df["Stock Quantity"] <= df["Minimum Required Level"])]
    
    if out_of_stock.empty and low_stock.empty:
        st.success("🎉 Excellent! All items are well-stocked.")
    
    if not out_of_stock.empty:
        st.warning(f"🚨 CRITICAL: {len(out_of_stock)} items are completely out of stock!")
        st.dataframe(out_of_stock[['Item Name', 'Stock Quantity']], use_container_width=True, hide_index=True)
        
    if not low_stock.empty:
        st.info(f"⚠️ LOW STOCK: {len(low_stock)} items are below reorder level and need attention.")
        st.dataframe(low_stock[['Item Name', 'Stock Quantity', 'Minimum Required Level']], use_container_width=True, hide_index=True)

# ==========================================
# STEP 7: PREDICTIVE NEEDS / RESTOCK PLANNING (FEATURE 3)
# ==========================================
def display_predictive_restock(df):
    st.subheader("🔮 Predictive Procurement Planning")
    predictive_df = df[df["Stock Quantity"] <= df["Minimum Required Level"]].copy()
    if predictive_df.empty:
        st.success("🟢 All stock buffers are verified secure for the current business cycle.")
    else:
        predictive_df["Suggested Restock Order"] = (predictive_df["Minimum Required Level"] * 2) - predictive_df["Stock Quantity"]
        st.dataframe(
            predictive_df[['Item Name', 'Stock Quantity', 'Minimum Required Level', 'Suggested Restock Order']],
            column_config={"Suggested Restock Order": st.column_config.NumberColumn("Urgent Restock Request", format="+%d units")},
            use_container_width=True, hide_index=True
        )


# ==========================================
# STEP 8: INVOICE MODULE 
# ==========================================
def run_quotation_module(df):
    st.subheader("🧾 Multi-Item Invoice Generator")
    valid_billing_df = df[df["Stock Quantity"] > 0].copy()
    if "cart" not in st.session_state: st.session_state.cart = []
    
    selected_items = st.multiselect("Select Items:", valid_billing_df["Item Name"].unique())
    if selected_items:
        for item in selected_items:
            qty = st.number_input(f"Qty for {item}", 1, 10, key=item)
            if st.button(f"Add {item}"):
                row = valid_billing_df[valid_billing_df["Item Name"] == item].iloc[0]
                st.session_state.cart.append({"name": item, "rate": row["Rate"], "qty": qty, "total": row["Rate"] * qty})
                st.rerun()

    if st.session_state.cart:
        cart_df = pd.DataFrame(st.session_state.cart)
        st.dataframe(cart_df)
        st.write(f"### Grand Total: Rs. {cart_df['total'].sum():,}")
        
        # New Clear Invoice Button
        if st.button("Clear Invoice / Start New"):
            st.session_state.cart = []
            st.rerun()

# ==========================================
# STEP 9: MAIN WEB CHATBOT ENGINE
# ==========================================
def main():
    faq_dataset = initialize_faq_dataset()
    idf_matrix = compute_idf(faq_dataset) 
    inventory_df = load_inventory_data()
    
    st.sidebar.title("⚙ Engine Dashboard")
    st.sidebar.metric(label="Total SKUs", value=len(inventory_df))
    st.sidebar.link_button("Chat with Support", url=get_whatsapp_link("03138492584"), use_container_width=True)
    
    st.title("⚙ Motorcycle Spare Parts AI Assistant")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📋 Inventory", "🚨 Alerts", "🔮 Restock", "🧾 Invoice"])
    
    with tab1:
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input("Ask about parts..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            response = find_best_faq_match(prompt, faq_dataset, idf_matrix, inventory_df)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
            
    with tab2: display_full_inventory(inventory_df)
    with tab3: display_short_out_alerts(inventory_df)
    with tab4: display_predictive_restock(inventory_df)
    with tab5: run_quotation_module(inventory_df)

if __name__ == "__main__":
    main()