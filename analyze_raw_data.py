import json
import os
import glob
from pathlib import Path
import numpy as np

def analyze_raw_field_data(date_dir):
    files = glob.glob(os.path.join(date_dir, "*.json"))
    report = {
        "summary": {"total_files": len(files), "sources": {}},
        "insights": [],
        "environmental_energy": None,
        "attention_surges": {},
        "crypto_health": {}
    }

    for f in files:
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
                source = data.get("source")
                report["summary"]["sources"][source] = report["summary"]["sources"].get(source, 0) + 1
                
                # Source-specific analysis
                if source == "open_meteo":
                    current = data.get("raw_data", {}).get("current", {})
                    cloud = current.get("cloud_cover", 0)
                    temp = current.get("temperature_2m", 0)
                    energy = 1.0 - (cloud / 100.0)
                    report["environmental_energy"] = {
                        "value": energy,
                        "state": "ACTION" if energy > 0.7 else "REFLECTION" if energy < 0.3 else "STABLE",
                        "temp": temp
                    }
                
                elif source == "wikipedia":
                    symbol = data.get("metadata", {}).get("symbol")
                    raw = data.get("raw_data", {})
                    items = raw.get("items", [])
                    if items:
                        views = [i["views"] for i in items[-7:]]
                        spike = (views[-1] - np.mean(views[:-1])) / (np.mean(views[:-1]) + 1e-10)
                        report["attention_surges"][symbol] = spike
                
                elif source == "coingecko":
                    symbol = data.get("metadata", {}).get("symbol")
                    raw = data.get("raw_data", {})
                    market = raw.get("market_data", {})
                    dev = raw.get("developer_data", {})
                    comm = raw.get("community_data", {})
                    
                    report["crypto_health"][symbol] = {
                        "price_change_24h": market.get("price_change_percentage_24h"),
                        "dev_activity": dev.get("commit_count_4_weeks"),
                        "sentiment_votes_up": raw.get("sentiment_votes_up_percentage"),
                        "reddit_active": comm.get("reddit_accounts_active_48h")
                    }
        except Exception as e:
            continue

    # Derive Cross-Context Insights
    if report["environmental_energy"]:
        ee = report["environmental_energy"]
        report["insights"].append(f"Environmental Field: {ee['state']} (Energy: {ee['value']:.2f}). {temp}°C in Bundjalung Country.")
    
    for sym, spike in report["attention_surges"].items():
        if spike > 0.2:
            report["insights"].append(f"Narrative Surge: {sym} Wikipedia attention is {spike:+.1%} above 7d mean. [SIGNAL: VOLATILITY BUILDING]")
        elif spike < -0.2:
            report["insights"].append(f"Narrative Cooling: {sym} interest dropping ({spike:+.1%}). [SIGNAL: FIELD CONSOLIDATION]")

    for sym, health in report["crypto_health"].items():
        if health["dev_activity"] and health["dev_activity"] > 50:
            report["insights"].append(f"Syntropic Crypto: {sym} has high developer activity ({health['dev_activity']} commits/4w). Stable backbone.")

    return report

if __name__ == "__main__":
    latest_dir = "/Users/sam/kindpath/kindpath-dfte/raw_field_data/2026-03-05"
    report = analyze_raw_field_data(latest_dir)
    print(json.dumps(report, indent=2))
