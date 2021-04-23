import scipy.stats
import sklearn.metrics
from ast import literal_eval
from analysis.create_plots import *
from analysis.calculate_baselines import calculate_freq_baseline, calculate_len_baseline, calculate_permutation_baseline


def extract_human_importance(dataset):
    with open("results/" + dataset + "_sentences.txt", "r") as f:
        sentences = f.read().splitlines()

    # split and lowercase
    tokens = [s.split(" ") for s in sentences]
    tokens = [[t.lower() for t in tokens] for tokens in tokens]

    human_importance = []
    with open("results/" + dataset + "_relfix_averages.txt", "r") as f:
        for line in f.read().splitlines():
            fixation_duration = np.fromstring(line, dtype=float, sep=',')
            human_importance.append(fixation_duration)

    return tokens, human_importance


# Importance type is either "saliency" or "attention"
def extract_model_importance(dataset, model, importance_type):
    lm_tokens = []
    lm_salience = []
    with open("results/" + dataset + "_" + model + "_" + importance_type + ".txt", "r") as f:
        for line in f.read().splitlines():
            tokens, heat = line.split("\t")

            tokens = list(literal_eval(tokens))
            salience = np.array(literal_eval(heat))

            # remove CLR and SEP tokens, this is an experimental choice
            lm_tokens.append(tokens[1:-1])
            salience = salience[1:-1]

            lm_salience.append(salience)

    return lm_tokens, lm_salience


def compare_importance(et_tokens, human_salience, lm_tokens, lm_salience, importance_type):
    count_tok_errors = 0

    spearman_correlations = []
    kendall_correlations = []
    mutual_information = []
    with open("results/" + corpus + "_" + model + "_" + importance_type + "_correlation_results.txt", "w") as outfile:
        outfile.write("Spearman\tKendall\tMutualInformation\n")
        for i, sentence in enumerate(et_tokens):
            if not len(et_tokens[i]) == len(lm_tokens[i]) == len(human_salience[i]) == len(lm_salience[i]):
                #                print("Tokenization Error:")
                #                print(et_tokens[i], lm_tokens[i])
                count_tok_errors += 1

            else:

                # Calculate the correlation
                spearman = scipy.stats.spearmanr(lm_salience[i], human_salience[i])[0]
                spearman_correlations.append(spearman)
                kendall = scipy.stats.kendalltau(lm_salience[i], human_salience[i])[0]
                kendall_correlations.append(kendall)
                mi_score = sklearn.metrics.mutual_info_score(lm_salience[i], human_salience[i])
                mutual_information.append(mi_score)
                outfile.write("{:.2f}\t{:.2f}\t{:.2f}\n".format(spearman, kendall, mi_score))
    print(corpus, model)
    print("Tokenization errors: ", count_tok_errors)
    print("Spearman Correlation Model: Mean, Stdev")
    mean_spearman = np.nanmean(np.asarray(spearman_correlations))
    std_spearman = np.nanstd(np.asarray(spearman_correlations))
    print(mean_spearman, std_spearman)

    print("\n\n\n")

    return mean_spearman, std_spearman


corpora = ["zuco", "geco"]
models = ["bert", "albert", "distil"]
types = ["saliency", "attention"]

baseline_results = pd.DataFrame(columns=('corpus', 'baseline_type', 'mean_correlation', 'std_correlation'))
results = pd.DataFrame(columns=('importance_type', 'corpus', 'model', 'mean_correlation', 'std_correlation'))
permutation_results = pd.DataFrame(
    columns=('importance_type', 'corpus', 'model', 'mean_correlation', 'std_correlation'))
for corpus in corpora:
    print(corpus)
    et_tokens, human_importance = extract_human_importance(corpus)

    # Length Baseline
    len_mean, len_std = calculate_len_baseline(et_tokens, human_importance)
    baseline_results = baseline_results.append(
        {'corpus': corpus, 'baselinetype': 'length', 'mean_correlation': len_mean, 'std_correlation': len_std},
        ignore_index=True)

    # Frequency Baseline
    pos_tags, frequencies = process_tokens(et_tokens)
    freq_mean, freq_std = calculate_freq_baseline(frequencies, human_importance)
    baseline_results = baseline_results.append(
        {'corpus': corpus, 'type': 'frequency', 'mean_correlation': freq_mean, 'std_correlation': freq_std},
        ignore_index=True)
    # Plots Human Importance
    # tag2humanimportance = calculate_saliency_by_wordclass(pos_tags, human_importance)
    # visualize_posdistribution(tag2humanimportance, "plots/" + corpus + "_human_wordclasses.png")
    for importance_type in types:
        print(importance_type)


        for model in models:
            lm_tokens, lm_importance = extract_model_importance(corpus, model, importance_type)

            # Model Correlation
            spearman_mean, spearman_std = compare_importance(et_tokens, human_importance, lm_tokens, lm_importance, importance_type)
            results = results.append(
                {'importance_type': importance_type, 'corpus': corpus, 'model': model, 'mean_correlation': spearman_mean, 'std_correlation': spearman_std},
                ignore_index=True)

            # Permutation Baseline
            spearman_mean, spearman_std = calculate_permutation_baseline(human_importance, lm_importance)
            permutation_results = permutation_results.append(
                {'importance_type': importance_type, 'corpus': corpus, 'model': model, 'mean_correlation': spearman_mean, 'std_correlation': spearman_std},
                ignore_index=True)

            # Plot Token-level analyses
     #       if corpus == "geco" and model == "bert" and importance_type == "saliency":
    #             lm_pos_tags, lm_frequencies = process_tokens(lm_tokens)
    #             visualize_frequencies(flatten(frequencies), flatten_saliency(human_importance), flatten(lm_frequencies),
    #                                   flatten_saliency(lm_importance), "plots/" + model + "_frequency.png")
    #             visualize_lengths(flatten(et_tokens), flatten_saliency(human_importance), flatten(lm_tokens),
    #                               flatten_saliency(lm_importance), "plots/" + model + "_length.png")
    #             tag2machineimportance = calculate_saliency_by_wordclass(lm_pos_tags, lm_importance)
    #             visualize_posdistribution(tag2machineimportance, "plots/" + model + "_wordclasses.png")
    #
    #             # Plot an example sentence
    #             i = 153
    #             visualize_sentence(i, et_tokens, human_importance, lm_importance, "plots/" + model + "_" + str(i) + ".png")
    with open("results/" + importance_type + "_all_results.txt", "w") as outfile:
        outfile.write("Model Importance: \n")
        outfile.write(results.to_latex())

        outfile.write("\n\nPermutation Baselines: \n")
        outfile.write(permutation_results.to_latex())

        outfile.write("\n\nLen-Freq Baselines: \n")
        outfile.write(baseline_results.to_latex())

        print(results)
        print()
        print(permutation_results)
        print()
        print(baseline_results)
        print()
