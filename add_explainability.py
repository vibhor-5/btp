import json

path = "notebooks/04_rnn_attention_models.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

captum_md = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## Advanced Explainability: Feature Importance with Captum (Integrated Gradients)\n",
  "Attention shows us *when* the model is looking, but Integrated Gradients tells us *which features* (temperature, sales, streak) are driving the stockout prediction at those times."
 ]
}

captum_code = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "!pip install -q captum\n",
  "from captum.attr import IntegratedGradients\n",
  "import matplotlib.pyplot as plt\n",
  "import seaborn as sns\n",
  "\n",
  "class ModelIGWrapper(nn.Module):\n",
  "    def __init__(self, model, model_name):\n",
  "        super().__init__()\n",
  "        self.model = model\n",
  "        self.model_name = model_name\n",
  "    \n",
  "    def forward(self, hist_num, hist_cat, future_num, future_cat):\n",
  "        batch = {\n",
  "            \"hist_num\": hist_num,\n",
  "            \"hist_cat\": hist_cat.long(),\n",
  "            \"future_num\": future_num,\n",
  "            \"future_cat\": future_cat.long()\n",
  "        }\n",
  "        if \"attention\" in self.model_name:\n",
  "            out = self.model(batch, return_attention=False)\n",
  "        else:\n",
  "            out = self.model(batch)\n",
  "        return out.mean(dim=1, keepdim=True)  # Avg over 7 days\n",
  "\n",
  "def plot_integrated_gradients(model_name=\"lstm_attention\", sample_idx=0):\n",
  "    model = model_factories[model_name]().to(device)\n",
  "    model.load_state_dict(torch.load(CKPT_DIR / f\"{model_name}.pt\", weights_only=False)[\"model_state\"])\n",
  "    model.eval()\n",
  "    \n",
  "    batch = next(iter(loaders[\"test\"]))\n",
  "    batch = {k: v.to(device) for k, v in batch.items()}\n",
  "    \n",
  "    wrapper = ModelIGWrapper(model, model_name)\n",
  "    ig = IntegratedGradients(wrapper)\n",
  "    \n",
  "    hist_num = batch[\"hist_num\"][sample_idx:sample_idx+1].requires_grad_()\n",
  "    hist_cat = batch[\"hist_cat\"][sample_idx:sample_idx+1]\n",
  "    future_num = batch[\"future_num\"][sample_idx:sample_idx+1]\n",
  "    future_cat = batch[\"future_cat\"][sample_idx:sample_idx+1]\n",
  "    \n",
  "    # We attribute only to hist_num for simplicity\n",
  "    attributions, delta = ig.attribute(inputs=hist_num,\n",
  "                                       additional_forward_args=(hist_cat, future_num, future_cat),\n",
  "                                       target=0, return_convergence_delta=True)\n",
  "    \n",
  "    attr_np = attributions.squeeze(0).cpu().detach().numpy()\n",
  "    \n",
  "    plt.figure(figsize=(14, 8))\n",
  "    sns.heatmap(attr_np, cmap=\"RdBu_r\", center=0, \n",
  "                xticklabels=HIST_NUM_COLS,\n",
  "                yticklabels=[f\"D-{CFG.history_len - i}\" for i in range(CFG.history_len)])\n",
  "    plt.title(f\"Integrated Gradients Feature Importance - {model_name}\")\n",
  "    plt.xlabel(\"Features\")\n",
  "    plt.ylabel(\"History Days\")\n",
  "    plt.tight_layout()\n",
  "    plt.show()\n",
  "\n",
  "    agg_attr = np.abs(attr_np).mean(axis=0)\n",
  "    plt.figure(figsize=(10, 5))\n",
  "    sns.barplot(x=agg_attr, y=HIST_NUM_COLS)\n",
  "    plt.title(f\"Overall Feature Importance (Magnitude) - {model_name}\")\n",
  "    plt.xlabel(\"Mean Absolute Attribution\")\n",
  "    plt.tight_layout()\n",
  "    plt.show()\n",
  "\n",
  "plot_integrated_gradients(\"lstm_attention\", sample_idx=5)\n"
 ]
}

shap_md = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## Advanced Explainability: SHAP (SHapley Additive exPlanations)\n",
  "We can also use SHAP's DeepExplainer on the PyTorch model to get global feature importance distributions."
 ]
}

shap_code = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "!pip install -q shap\n",
  "import shap\n",
  "\n",
  "def plot_shap(model_name=\"lstm_attention\"):\n",
  "    model = model_factories[model_name]().to(device)\n",
  "    model.load_state_dict(torch.load(CKPT_DIR / f\"{model_name}.pt\", weights_only=False)[\"model_state\"])\n",
  "    model.eval()\n",
  "    \n",
  "    wrapper = ModelIGWrapper(model, model_name)\n",
  "    \n",
  "    # 1. Prepare background data for DeepExplainer\n",
  "    bg_batch = next(iter(loaders[\"train\"]))\n",
  "    bg_num = bg_batch[\"hist_num\"][:50].to(device)  # 50 samples for background\n",
  "    bg_cat = bg_batch[\"hist_cat\"][:50].to(device)\n",
  "    bg_f_num = bg_batch[\"future_num\"][:50].to(device)\n",
  "    bg_f_cat = bg_batch[\"future_cat\"][:50].to(device)\n",
  "    \n",
  "    # We need to wrap it so it only takes 1 tensor input for SHAP's DeepExplainer\n",
  "    class SingleInputWrapper(nn.Module):\n",
  "        def __init__(self, wrapper_model, hist_cat, future_num, future_cat):\n",
  "            super().__init__()\n",
  "            self.wrapper_model = wrapper_model\n",
  "            self.hist_cat = hist_cat\n",
  "            self.future_num = future_num\n",
  "            self.future_cat = future_cat\n",
  "        def forward(self, hist_num):\n",
  "            # Repeat the static/categorical features to match batch size of hist_num\n",
  "            b = hist_num.shape[0]\n",
  "            return self.wrapper_model(hist_num, self.hist_cat[:b], self.future_num[:b], self.future_cat[:b])\n",
  "            \n",
  "    # 2. Test data\n",
  "    test_batch = next(iter(loaders[\"test\"]))\n",
  "    test_num = test_batch[\"hist_num\"][:20].to(device)\n",
  "    \n",
  "    shap_model = SingleInputWrapper(wrapper, test_batch[\"hist_cat\"].to(device), test_batch[\"future_num\"].to(device), test_batch[\"future_cat\"].to(device))\n",
  "    \n",
  "    # 3. Create explainer\n",
  "    explainer = shap.DeepExplainer(shap_model, bg_num)\n",
  "    shap_values = explainer.shap_values(test_num)\n",
  "    \n",
  "    # Reshape for plotting\n",
  "    shap_values_flat = shap_values.reshape(test_num.shape[0], -1)\n",
  "    test_num_flat = test_num.cpu().numpy().reshape(test_num.shape[0], -1)\n",
  "    \n",
  "    feature_names = []\n",
  "    CFG.history_len = 14\n",
  "    for d in range(CFG.history_len):\n",
  "        for col in HIST_NUM_COLS:\n",
  "            feature_names.append(f\"D-{CFG.history_len-d}_{col}\")\n",
  "            \n",
  "    plt.figure(figsize=(10, 8))\n",
  "    shap.summary_plot(shap_values_flat, test_num_flat, feature_names=feature_names, max_display=15)\n",
  "    \n",
  "plot_shap(\"lstm_attention\")\n"
 ]
}

nb["cells"].extend([captum_md, captum_code, shap_md, shap_code])

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
