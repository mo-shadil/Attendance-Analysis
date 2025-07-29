import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

print("STUDENT ATTENDANCE ANALYSIS REPORT")
print("=" * 60)

# 1. Load the Excel file
print("\nLoading attendance data...")
df = pd.read_excel("Attendance Table Structure New.xlsx", header=1)
print(f"Loaded {len(df)} attendance records with {len(df.columns)} columns.")

# 2. Calculate daily attendance per student
print("\nCalculating daily attendance for each student...")
def calc_daily_attendance(row):
    hours = [row[f"Hour{i}"] for i in range(1, 10)]
    present = any(h == 1 for h in hours if h != 'F' and pd.notnull(h))
    all_null_or_f = all((pd.isnull(h) or h == 'F') for h in hours)
    if all_null_or_f:
        return np.nan
    return 1 if present else 0

df["IsPresent"] = df.apply(calc_daily_attendance, axis=1)

# 3. Prepare melted DataFrame
melted = df[["SemesterYearstudentID", "AttendanceDate", "IsPresent"]].copy()
melted.rename(columns={"SemesterYearstudentID": "StudentID", "AttendanceDate": "Date"}, inplace=True)

# 4. Convert Date to datetime
melted["Date"] = pd.to_datetime(melted["Date"], dayfirst=True, errors='coerce')

# 5. Pivot table: StudentID × Date
print("Creating attendance matrix...")
pivot_df = melted.pivot(index='StudentID', columns='Date', values='IsPresent')
pivot_df = pivot_df.fillna(0).astype(int)

total_days = pivot_df.shape[1]
pivot_df['TotalAbsences'] = (pivot_df == 0).sum(axis=1)
pivot_df['AttendancePercent'] = ((pivot_df == 1).sum(axis=1) / total_days) * 100

print(f"Total school days analyzed: {total_days}")
print(f"Total students analyzed: {len(pivot_df)}")

# 6. Calculate consecutive absences
print("Analyzing consecutive absences...")
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
pivot_df["MaxConsecutiveAbsences"] = pivot_df.index.map(consecutive_absences)

# 7. Z-score anomaly detection
pivot_df["ZScore_Attendance"] = zscore(pivot_df["AttendancePercent"]).round(2)
pivot_df["ZScore_Anomaly"] = pivot_df["ZScore_Attendance"].apply(
    lambda z: "Unusually Low Attendance" if z < -1.0 else "Normal Attendance Pattern"
)

# 8. Isolation Forest anomaly detection
features = pivot_df[["AttendancePercent", "TotalAbsences", "MaxConsecutiveAbsences"]].dropna()
if features.empty:
    pivot_df["Anomaly_IForest"] = "Not Available"
else:
    model = IsolationForest(contamination=0.2, random_state=42)
    preds = model.fit_predict(features)
    pivot_df["Anomaly_IForest"] = pd.Series(preds, index=features.index).map({-1: "Irregular Attendance Pattern", 1: "Normal Attendance Pattern"})
    # Fill missing with Not Available for students not in features
    pivot_df["Anomaly_IForest"].fillna("Not Available", inplace=True)

# 9. Attendance risk categorization
def categorize_risk(pct):
    if pct >= 90:
        return "Excellent"
    elif pct >= 80:
        return "Good"
    elif pct >= 70:
        return "Warning"
    else:
        return "High Risk"

pivot_df["RiskCategory"] = pivot_df["AttendancePercent"].apply(categorize_risk)

# 10. Show final results
print("\n" + "="*60)
print("FINAL ATTENDANCE RISK REPORT")
print("="*60)

summary_df = pivot_df[[
    "AttendancePercent",
    "TotalAbsences",
    "MaxConsecutiveAbsences",
    "ZScore_Attendance",
    "ZScore_Anomaly",
    "Anomaly_IForest",
    "RiskCategory"
]].copy()
summary_df["AttendancePercent"] = summary_df["AttendancePercent"].round(1)
summary_df.columns = [
    "Attendance %",
    "Total Absences",
    "Max Consecutive Absences",
    "ZScore Attendance",
    "ZScore Anomaly",
    "Anomaly Forest",
    "Risk Level"
]
summary_df = summary_df.sort_values("Attendance %")

print(f"\nSUMMARY STATISTICS:")
print(f"   • Total Students: {len(summary_df)}")
print(f"   • High Risk: {len(summary_df[summary_df['Risk Level'] == 'High Risk'])}")
print(f"   • Warning: {len(summary_df[summary_df['Risk Level'] == 'Warning'])}")
print(f"   • Good: {len(summary_df[summary_df['Risk Level'] == 'Good'])}")
print(f"   • Excellent: {len(summary_df[summary_df['Risk Level'] == 'Excellent'])}")

print(f"\nSTUDENTS REQUIRING IMMEDIATE ATTENTION (High Risk):")
high_risk_students = summary_df[summary_df['Risk Level'] == 'High Risk']
if len(high_risk_students) > 0:
    for idx, (student_id, row) in enumerate(high_risk_students.head(10).iterrows(), 1):
        print(f"   {idx}. Student ID: {student_id}")
        print(f"      Attendance: {row['Attendance %']}% | Total Absences: {row['Total Absences']} | Longest Absence Streak: {row['Max Consecutive Absences']}")
        print(f"      ZScore: {row['ZScore Attendance']} | ZScore Anomaly: {row['ZScore Anomaly']} | Anomaly Forest: {row['Anomaly Forest']}")
        print()
else:
    print("   No students in high risk category.")

print(f"\nSTUDENTS WITH WARNING (Need Monitoring):")
warning_students = summary_df[summary_df['Risk Level'] == 'Warning']
if len(warning_students) > 0:
    for idx, (student_id, row) in enumerate(warning_students.head(5).iterrows(), 1):
        print(f"   {idx}. Student ID: {student_id} - Attendance: {row['Attendance %']}% | ZScore: {row['ZScore Attendance']} | ZScore Anomaly: {row['ZScore Anomaly']} | Anomaly Forest: {row['Anomaly Forest']}")
else:
    print("   No students in warning category.")

print(f"\nTOP PERFORMERS (Excellent Attendance):")
excellent_students = summary_df[summary_df['Risk Level'] == 'Excellent']
if len(excellent_students) > 0:
    for idx, (student_id, row) in enumerate(excellent_students.head(5).iterrows(), 1):
        print(f"   {idx}. Student ID: {student_id} - Attendance: {row['Attendance %']}%")
else:
    print("   No students with excellent attendance.")

# 11. Save output to Excel
print(f"\nSaving detailed report...")
summary_df.to_excel("final_attendance_report.xlsx")
print("Report saved to: final_attendance_report.xlsx")

print(f"\nRISK LEVEL DEFINITIONS:")
print("   Excellent: 90%+ attendance - Student is performing very well")
print("   Good: 80-89% attendance - Student has acceptable attendance")
print("   Warning: 70-79% attendance - Student needs attention and monitoring")
print("   High Risk: <70% attendance - Student requires immediate intervention")

print(f"\nRECOMMENDED ACTIONS:")
print("   • High Risk: Schedule parent meeting, implement intervention plan")
print("   • Warning: Monitor closely, send warning letter, check for underlying issues")
print("   • Good: Continue current approach, provide positive reinforcement")
print("   • Excellent: Recognize achievement, consider leadership opportunities")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
