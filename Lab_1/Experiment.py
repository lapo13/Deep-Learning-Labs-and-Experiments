from matplotlib import pyplot as plt
import json
import datetime, os



class _Logger:
    def __init__(self, experiment_name="experiment"):
        self.logs = {}
        self.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"logs/{experiment_name}_{self.start_time}"
        os.makedirs(self.save_dir, exist_ok=True)

    def log(self, key, value):
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(value)

    def save(self, filename="metrics.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.logs, f, indent=4) 
        print(f"Log salvati in: {path}")

    def plot(self, keys, save_plot=True):
     if isinstance(keys, str):
          keys = [keys]

     plt.figure(figsize=(10, 6))
     
     keys_found = []
     for key in keys:
          if key in self.logs:
               plt.plot(self.logs[key], marker='o', linestyle='-', label=key)
               keys_found.append(key)
          else:
               print(f"Avviso: La chiave '{key}' non esiste nei log e verrà saltata.")
     
     if not keys_found:
          plt.close()
          return


     title_str = " vs ".join(keys_found)
     if len(keys_found) == 1:
          title_str = keys_found[0]
     plt.title(f"Andamento: {title_str}")
     plt.xlabel('Epochs')
     plt.ylabel('Value')
     plt.grid(True, linestyle='--', alpha=0.7)

     if len(keys_found) > 1:
          plt.legend()

     if save_plot:
          filename = f"{'_vs_'.join(keys_found)}.png".replace(" ", "_")
          plot_path = os.path.join(self.save_dir, filename)
          plt.savefig(plot_path)
          print(f"Grafico salvato in: {plot_path}")
     
     plt.show()
     plt.close()

class experiment:
    def __init__(self, trainer, data_module, name="Experiment"):
        self.trainer = trainer
        self.DM = data_module
        self.logger = _Logger(experiment_name=name)

    def run(self, epochs, early_stopping=False, patience=5, tolerance=1e-3):
        print(f"Starting experiment: {self.logger.save_dir}")

        dl_train, dl_val, dl_test = self.DM.split_data()

        print("Training model...")
        self.trainer.fit(
            dl_train, 
            dl_val, 
            epochs=epochs, 
            patience=patience if early_stopping else float('inf'), 
            tolerance=tolerance
        )

        for key, values in self.trainer.history.items():
            for v in values:
                self.logger.log(key, v)
        
        print("Evaluating...")
        report, _ = self.trainer.evaluate(dl_test, test_mode=True)
        
        report_path = os.path.join(self.logger.save_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        self.logger.save()
        print(f"Experiment completed. Results in: {self.logger.save_dir}")
        print("\n\nClassification Report:\n", report)

    def get_results(self):
        return self.logger.logs

    def plot_results(self, groups=None):
        if groups is None:
            groups = list(self.logger.logs.keys())

        for group in groups:
            self.logger.plot(group)