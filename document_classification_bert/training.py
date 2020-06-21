from tqdm import tqdm
import torch.nn as nn


def train_epoch( model, data_loader, loss_fn, optimizer, device, scheduler, training_inputs):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for i, d in tqdm(enumerate(data_loader), total= len(data_loader)):
    input_ids = d['ids'].to(device)
    attention_mask = d['mask'].to(device)
    targets = d['label'].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / len(training_inputs), np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, validation_inputs):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["ids"].to(device)
      attention_mask = d["mask"].to(device)
      targets = d["label"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / len(validation_inputs), np.mean(losses)









