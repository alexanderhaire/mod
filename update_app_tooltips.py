
import os

file_path = "c:\\Users\\alexh\\Downloads\\mod\\app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = 0

for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue

    # 1. Update Aggregation
    if "chart_data = data_df.groupby('TransactionDate').agg({" in line:
        # Check if next lines match what we expect
        if "'AvgCost': 'mean'," in lines[i+1]:
            # Replace the whole block
            indent = line.split("chart_data")[0]
            new_lines.append(f"{indent}chart_data = data_df.groupby('TransactionDate').agg({{\n")
            new_lines.append(f"{indent}    'AvgCost': 'mean',\n")
            new_lines.append(f"{indent}    'LandedCost': 'mean',\n")
            new_lines.append(f"{indent}    'TransactionCount': 'sum',\n")
            new_lines.append(f"{indent}    'Quantity': 'sum'\n")
            new_lines.append(f"{indent}}}).reset_index()\n")
            new_lines.append("\n")
            new_lines.append(f"{indent}chart_data['AvgCost'] = chart_data['AvgCost'].astype(float)\n")
            new_lines.append(f"{indent}chart_data['LandedCost'] = chart_data['LandedCost'].astype(float)\n")
            new_lines.append(f"{indent}chart_data['TransactionCount'] = chart_data['TransactionCount'].astype(float)\n")
            new_lines.append(f"{indent}chart_data['Quantity'] = chart_data['Quantity'].astype(float)\n")
            
            # Skip old lines: 
            # 1: 'AvgCost'...
            # 2: 'TransactionCount'...
            # 3: 'Quantity'...
            # 4: }).reset_index()
            # 5: empty (maybe)
            # 6: chart_data['AvgCost']...
            # 7: chart_data['TransactionCount']...
            # 8: chart_data['Quantity']...
            skip = 8 # Ensure we skip the old casting lines too
            continue

    # 2. Update Line Tooltip
    if "tooltip=['TransactionDate', 'AvgCost', 'TransactionCount']" in line:
        indent = line.split("tooltip")[0]
        new_lines.append(f"{indent}tooltip=[\n")
        new_lines.append(f"{indent}    alt.Tooltip('TransactionDate', title='Date', format='%Y-%m-%d'),\n")
        new_lines.append(f"{indent}    alt.Tooltip('AvgCost', title='Delivered Cost', format='$,.4f'),\n")
        new_lines.append(f"{indent}    alt.Tooltip('LandedCost', title='Landed Cost', format='$,.4f'),\n")
        new_lines.append(f"{indent}    alt.Tooltip('Quantity', title='Quantity', format=',.0f')\n")
        new_lines.append(f"{indent}]\n")
        continue

    # 3. Update Bar Tooltip
    if "tooltip=['TransactionDate', 'Quantity']" in line:
        indent = line.split("tooltip")[0]
        new_lines.append(f"{indent}tooltip=[\n")
        new_lines.append(f"{indent}    alt.Tooltip('TransactionDate', title='Date', format='%Y-%m-%d'),\n")
        new_lines.append(f"{indent}    alt.Tooltip('AvgCost', title='Delivered Cost', format='$,.4f'),\n")
        new_lines.append(f"{indent}    alt.Tooltip('LandedCost', title='Landed Cost', format='$,.4f'),\n")
        new_lines.append(f"{indent}    alt.Tooltip('Quantity', title='Quantity', format=',.0f')\n")
        new_lines.append(f"{indent}]\n")
        continue

    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Updates applied.")
