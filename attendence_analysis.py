import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# 📥 1. Load the Excel file (update the filename as needed)
df = pd.read_excel("Attendance Table Structure New.xlsx", header=1)
print("Original columns in the Excel file:", df.columns.tolist())

# 🔍 2. Calculate daily attendance per student (1 if present in any hour, 0 if absent in all, np.nan if all NULL/F)
def calc_daily_attendance(row):
    hours = [row[f"Hour{i}"] for i in range(1, 10)]
    # Treat 'F' as not counted (frozen), NULL as np.nan
    present = any(h == 1 for h in hours if h != 'F' and pd.notnull(h))
    all_null_or_f = all((pd.isnull(h) or h == 'F') for h in hours)
    if all_null_or_f:
        return np.nan
    return 1 if present else 0

df["IsPresent"] = df.apply(calc_daily_attendance, axis=1)

# 🧼 3. Prepare melted DataFrame: StudentID, Date, IsPresent
melted = df[["SemesterYearstudentID", "AttendanceDate", "IsPresent"]].copy()
melted.rename(columns={"SemesterYearstudentID": "StudentID", "AttendanceDate": "Date"}, inplace=True)

# 🧼 4. Convert Date to datetime
melted["Date"] = pd.to_datetime(melted["Date"], dayfirst=True, errors='coerce')

# 🧮 5. Recalculate pivot table: StudentID × Date
pivot_df = melted.pivot(index='StudentID', columns='Date', values='IsPresent')
pivot_df = pivot_df.fillna(0).astype(int)

# 📊 6. Calculate total absences and attendance percent
total_days = pivot_df.shape[1]
pivot_df['TotalAbsences'] = (pivot_df == 0).sum(axis=1)
pivot_df['AttendancePercent'] = ((pivot_df == 1).sum(axis=1) / total_days) * 100

# 📉 7. Calculate consecutive absences
melted.sort_values(by=["StudentID", "Date"], inplace=True)
consecutive_absences = {}
for student_id, group in melted.groupby("StudentID"):
    count = 0
    max_streak = 0
    for present in group["IsPresent"]:
        if present == 0:
            count += 1
            max_streak = max(max_streak, count)
        else:
            count = 0
    consecutive_absences[student_id] = max_streak
pivot_df["ConsecutiveAbsences"] = pivot_df.index.map(consecutive_absences)

# 📉 8. Z-score anomaly detection
pivot_df["ZScore_Attendance"] = zscore(pivot_df["AttendancePercent"])
pivot_df["ZScore_Anomaly"] = pivot_df["ZScore_Attendance"].apply(lambda z: -1 if z < -1.0 else 1)

# 🧪 9. ML Anomaly Detection (safe check)
features = pivot_df[["AttendancePercent", "TotalAbsences", "ConsecutiveAbsences"]].dropna()
if features.empty:
    print("🚫 No valid data available for ML anomaly detection.")
    pivot_df["Anomaly_IForest"] = "N/A"
else:
    model = IsolationForest(contamination=0.2, random_state=42)
    pivot_df["Anomaly_IForest"] = model.fit_predict(features)

# 🚦 10. Risk level tagging
def categorize_risk(pct):
    if pct >= 90:
        return "✅ Safe"
    elif pct >= 75:
        return "⚠️ Warning"
    else:
        return "🚨 High Risk"

pivot_df["RiskCategory"] = pivot_df["AttendancePercent"].apply(categorize_risk)

# ✅ 11. Show final results
print("\n✅ Final Attendance Risk Report:")
print(pivot_df[[
    "AttendancePercent",
    "TotalAbsences",
    "ConsecutiveAbsences",
    "ZScore_Attendance",
    "ZScore_Anomaly",
    "Anomaly_IForest",
    "RiskCategory"
]])

# 💾 12. Save output to Excel
pivot_df.to_excel("final_attendance_report.xlsx")
print("\n📁 Report saved to: final_attendance_report.xlsx")
