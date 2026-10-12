import re
import pandas as pd

with open('aula3/logs/fe_train.log', 'r', encoding='utf-16le', errors='ignore') as f:
    text = f.read()

# Strip ANSI codes and normalize newlines
clean_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
lines = [l.strip() for l in clean_text.replace('\r', '\n').split('\n') if 'val_loss' in l and '606/606' in l]

data = []
for i, line in enumerate(lines):
    # Extract time in seconds
    t_match = re.search(r'(\d+)s\s+\d+ms/step', line)
    time_s = int(t_match.group(1)) if t_match else None
    
    # Extract metrics
    acc = float(re.search(r'accuracy:\s*([\d\.]+)', line).group(1))
    auc = float(re.search(r'auc:\s*([\d\.]+)', line).group(1))
    loss = float(re.search(r'loss:\s*([\d\.]+)', line).group(1))
    val_acc = float(re.search(r'val_accuracy:\s*([\d\.]+)', line).group(1))
    val_auc = float(re.search(r'val_auc:\s*([\d\.]+)', line).group(1))
    val_loss = float(re.search(r'val_loss:\s*([\d\.]+)', line).group(1))
    
    data.append({
        'epoch': i + 1,
        'loss': loss,
        'accuracy': acc,
        'auc': auc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_auc': val_auc,
        'time_s': time_s
    })

df = pd.DataFrame(data)
df.to_csv('aula3/results/history_fe.csv', index=False)
print(f'Total epochs parsed: {len(df)}')
print(df.to_string(index=False))

total_epoch_time = df['time_s'].sum()
print(f'\nTotal sum of epoch times: {total_epoch_time} s ({total_epoch_time/60:.2f} min)')
min_val_loss_idx = df['val_loss'].idxmin()
print(f'Min val_loss at index: {min_val_loss_idx} (0-indexed) -> Epoch {min_val_loss_idx + 1} (1-indexed)')
print('\nBest epoch row (1-indexed Epoch {}):'.format(min_val_loss_idx + 1))
print(df.iloc[min_val_loss_idx].to_dict())
