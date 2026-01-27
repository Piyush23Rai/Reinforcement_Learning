import random, string
from datetime import datetime, timedelta
import csv

random.seed(7)

MERCHANTS = ["M102","M311","M450","M522","M601"]
COUNTRIES = ["US","GB","DE","FR","SG","IN"]
CURRENCIES = {"US":"USD","GB":"GBP","DE":"EUR","FR":"EUR","SG":"SGD","IN":"INR"}
ERRORS = [("",0.70), ("GATEWAY_TIMEOUT",0.07), ("ISSUER_DO_NOT_HONOR",0.12), ("INSUFFICIENT_FUNDS",0.06), ("FRAUD_SUSPECTED",0.05)]
AVS = ["AVS_MATCH","AVS_MISMATCH","AVS_UNAVAILABLE"]
THREEDS = ["FRICTIONLESS","CHALLENGE","NOT_ATTEMPTED"]

def weighted_choice(pairs):
    r = random.random()
    s = 0.0
    for item, w in pairs:
        s += w
        if r <= s:
            return item
    return pairs[-1][0]

def txn_id():
    return "T" + "".join(random.choices(string.digits, k=6))

def gen_row(t):
    country = random.choice(COUNTRIES)
    currency = CURRENCIES[country]
    amount = round(random.choice([9.99, 12.50, 19.00, 49.99, 99.00, 199.00, 299.00]) * (1 + random.random()*0.1), 2)
    merchant = random.choice(MERCHANTS)

    err = weighted_choice(ERRORS)
    fraud_score = random.randint(1, 100)
    avs = random.choice(AVS)
    three_ds = random.choice(THREEDS)

    if err == "":
        auth_result = "APPROVED"
        error_code = ""
        issuer_resp = ""
    elif err == "GATEWAY_TIMEOUT":
        auth_result = "FAILED"
        error_code = "GATEWAY_TIMEOUT"
        issuer_resp = ""
    else:
        auth_result = "DECLINED"
        error_code = err
        issuer_resp = random.choice(["05","51","57","62"]) if err != "FRAUD_SUSPECTED" else "59"

    # simplistic chargeback proxy (approved + high risk)
    chargeback_flag = 1 if (auth_result=="APPROVED" and fraud_score > 80 and three_ds != "CHALLENGE") else 0

    return [t.isoformat()+"Z", merchant, txn_id(), amount, currency, country, "card", auth_result, error_code, issuer_resp, fraud_score, avs, three_ds, chargeback_flag]

def generate_csv(path="fintech_events.csv", n=5000):
    start = datetime(2026,1,26,9,0,0)
    with open(path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_time","merchant_id","txn_id","amount","currency","country","payment_method","auth_result","error_code","issuer_response","fraud_score","avs_result","three_ds","chargeback_flag"])
        t = start
        for _ in range(n):
            t += timedelta(seconds=random.randint(5, 45))
            w.writerow(gen_row(t))

generate_csv()
print("Wrote fintech_events.csv")
