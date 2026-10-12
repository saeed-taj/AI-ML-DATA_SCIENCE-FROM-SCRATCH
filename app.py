import ast
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Electronics Recommendation System", page_icon="💻", layout="wide"
)

st.title(" Electronics Recommendation & Market Basket Analysis System")
st.markdown(
    "Assignment #01 - Applied Machine Learning | Association Rule Mining"
)


# 1. Load and Cache Data
@st.cache_data
def load_data():
  df = pd.read_csv("dataset.csv")
  # Convert string representation of lists back to actual Python lists
  transactions = df["Items"].apply(ast.literal_eval).tolist()
  return df, transactions


df, transactions = load_data()

# 2. Sidebar Controls (Threshold Playground & Parameters)
st.sidebar.header("⚙️ Model Parameters")
min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.01,
    max_value=0.20,
    value=0.04,
    step=0.01,
    help="Adjust minimum support threshold for Apriori",
)
min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
    help="Adjust minimum confidence threshold for rules",
)

# 3. Data Preprocessing & Apriori Engine Execution
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(
    df_encoded, min_support=min_support, use_colnames=True
)
rules = pd.DataFrame()
if not frequent_itemsets.empty:
  rules = association_rules(
      frequent_itemsets, metric="confidence", min_threshold=min_confidence
  )

# 4. Streamlit Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " Dataset Overview",
    " Frequent Itemsets",
    " Rules Explorer",
    " Smart Basket Recommendations",
])

#  TAB 1: OVERVIEW 
with tab1:
  st.subheader("Dataset Summary Statistics")
  col1, col2, col3 = st.columns(3)
  col1.metric("Total Transactions", len(df))
  col2.metric("Unique Items", len(te.columns_))
  col3.metric("Discovered Rules", len(rules))

  st.dataframe(df.head(10), use_container_width=True)

  st.subheader("Item Frequency Distribution")
  item_counts = (
      pd.Series([item for sublist in transactions for item in sublist])
      .value_counts()
      .reset_index()
  )
  item_counts.columns = ["Item", "Frequency"]
  st.bar_chart(item_counts.set_index("Item"))

#  TAB 2: FREQUENT ITEMSETS 
with tab2:
  st.subheader("Generated Frequent Itemsets (Apriori)")
  st.write(
      f"Showing itemsets meeting Minimum Support: **{min_support}**"
  )
  if not frequent_itemsets.empty:
    st.dataframe(
        frequent_itemsets.sort_values(by="support", ascending=False),
        use_container_width=True,
    )
  else:
    st.warning(
        "No frequent itemsets found. Try lowering the Minimum Support slider in"
        " the sidebar."
    )

#  TAB 3: RULES EXPLORER 
with tab3:
  st.subheader("Association Rules Explorer")
  if not rules.empty:
    # Format frozensets to readable strings for display
    display_rules = rules.copy()
    display_rules["antecedents"] = display_rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )
    display_rules["consequents"] = display_rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    st.dataframe(
        display_rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
            ]
        ].sort_values(by="confidence", ascending=False),
        use_container_width=True,
    )
  else:
    st.warning(
        "No association rules generated under current thresholds. Lower support"
        " or confidence."
    )

#  TAB 4: SMART BASKET RECOMMENDATION ENGINE 
with tab4:
  st.subheader(" Interactive Smart Basket")
  st.write("Select items currently in your shopping cart:")

  all_available_items = sorted(list(te.columns_))
  selected_items = st.multiselect(
      "Choose items:", all_available_items, default=[all_available_items[0]]
  )

  if selected_items and not rules.empty:
    st.markdown("")
    st.subheader("Recommendations for You:")

    recommendations = []
    selected_set = set(selected_items)

    for index, row in rules.iterrows():
      antecedent = set(row["antecedents"])
      consequent = set(row["consequents"])

      # Check if rule antecedent is a subset of user selected items
      if antecedent.issubset(selected_set):
        # Prevent recommending items already in user basket
        new_recs = consequent - selected_set
        for rec in new_recs:
          recommendations.append({
              "Recommended Item": rec,
              "Confidence": row["confidence"],
              "Support": row["support"],
              "Lift": row["lift"],
              "Rule": (
                  f"{{{', '.join(antecedent)}}} $\\rightarrow$ {{{rec}}}"
              ),
          })

    if recommendations:
      rec_df = (
          pd.DataFrame(recommendations)
          .sort_values(by="Confidence", ascending=False)
          .drop_duplicates(subset=["Recommended Item"])
      )

      for _, r in rec_df.iterrows():
        with st.container():
          st.success(
              f"**Recommended:** `{r['Recommended Item']}` (Confidence:"
              f" `{r['Confidence']:.2f}` | Support: `{r['Support']:.2f}`)"
          )
          st.caption(f" **Why?** Recommended because rule {r['Rule']} matched.")
    else:
      st.info(
          "No direct recommendations found for this specific combination under"
          " current rules."
      )
  else:
    st.info(
        "Please select at least one item in your smart basket or ensure rules"
        " are generated."
    )