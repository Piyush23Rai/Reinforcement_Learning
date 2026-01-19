# Toy Semantic Search Actor–Critic (Streamlit)

A class-ready demo showing Actor–Critic roles in a production-like workflow:

- User enters query + constraints
- Actor chooses retrieval strategy
- Critic predicts expected outcome V(s)
- Reward comes from manual feedback or a toy KPI proxy
- Update uses Advantage A = r - V(s)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
