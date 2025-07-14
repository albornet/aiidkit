import os
import datetime
import numpy as np
import pandas as pd
import src.constants as constants
import matplotlib.pyplot as plt
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from src.data.data_utils import get_valid_categories

csts = constants.ConstantsNamespace()
PLOT_DIR = os.path.join(csts.RESULT_DIR_PATH, "explore_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def get_survival_plot_dict(
    pat_inf: pd.DataFrame,  # patient infection dataframe
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """ Dictionary to generate different survival analysis plots
    """
    survival_plot_dict = {
        "ALL": {"All patients": None},
        # "MDR": {
        #     "MDR infections": {"rp_mdr_0": ["Yes"]},
        #     "no-MDR infections": {"rp_mdr_0": ["No"]},
        # },
        # "INFTYPE": {
        #     "Bacteria": {"bact_0": get_valid_categories(pat_inf, "bact_0")},
        #     "Virus": {"virus_0": get_valid_categories(pat_inf, "virus_0")},
        #     "Fungus": {"fungi_0": get_valid_categories(pat_inf, "fungi_0")},
        #     "Parasite": {"parasit_0": get_valid_categories(pat_inf, "parasit_0")},
        #     "All infections": None,
        # },
        # "INFSITE": {
        #     "Urinary tract": {"infsite_0": ["Urinary tract"]},
        #     "Blood": {"infsite_0": ["Blood"]},
        #     "Respiratory tract": {"infsite_0": ["RT"]},
        #     "Gastrointestinal tract": {"infsite_0": ["GI"]},
        #     "Mucocutaneous": {"infsite_0": ["Mucocutaneous"]},
        #     "All sites": None,
        # },
        # "VIRUS": {
        #     "Herpesviridae ": {"virus_0": ["CMV", "HSV", "EBV", "VZV", "HHV6", "HHV8"]},
        #     "Respiratory": {"virus_0": ["Influenza", "RSV", "Parainfluenza", "Rhinovirus", "Metapneumovirus", "Adenovirus"]},
        #     "Polyomaviruses": {"virus_0": ["BKV", "JCV"]},
        #     "Hepatitis": {"virus_0": ["HCV", "HBV"]},
        # },
        # "BACT": {
        #     "E._coli": {"bact_0": ["E. coli"]},
        #     "Other enterobacteriales": {"bact_0": ["Klebsiella sp", "Enterobacter", "Other enterobacteria"]},
        #     "Enterococcus": {"bact_0": ["Enterococcus"]},
        #     "Klebsiella": {"bact_0": ["Klebsiella sp"]},
        #     "Staphylococcus": {"bact_0": ["Staph aureus", "St. coagulase negative", "MSSA", "MRSA"]},
        #     "Pseudomonas": {"bact_0": ["Pseudomonas aeruginosa"]},
        # },
        # "URINARY": {
        #     "Enterobacteriaceae" : {"infsite_0": ["Urinary tract"], "bact_0": ["E. coli", "Klebsiella sp", "Enterobacter", "Other enterobacteria"]},
        #     "Enterococcus": {"infsite_0": ["Urinary tract"], "bact_0": ["Enterococcus"]},
        #     "Pseudomonas aeruginosa": {"infsite_0": ["Urinary tract"], "bact_0": ["Pseudomonas aeruginosa"]},
        # },
        # "DONOR": {
        #     "Potentially donor related": {"donorrelid": ["Yes"]},
        #     "Not donor related": {"donorrelid": ["No"]},
        # },
        # "ISRED": {
        #     "With immunosuppression reduction": {"isred": ["Yes"]},
        #     "Without immunosuppression reduction": {"isred": ["No"]},
        # },
    }

    return survival_plot_dict


def generate_survival_analysis_plots(
    pat_inf: pd.DataFrame,  # patient infection dataframe
    pat_stop: pd.DataFrame,  # patient censoring dataframe
    kid_bl: pd.DataFrame,  # organ baseline dataframe
) -> None:
    """ ...
    """
    # Generate various survival analysis plots
    survival_plot_dict = get_survival_plot_dict(pat_inf)
    for title_tag, infection_constraint_dict in survival_plot_dict.items():
        kmf_dict = {}
        for constraint_name, infection_constraint in tqdm(
            iterable=infection_constraint_dict.items(),
            desc=f"Computing survival analysis for {title_tag}",
        ):
            kmf = compute_infection_survival_analysis_results(
                pat_inf=pat_inf,
                pat_stop=pat_stop,
                kid_bl=kid_bl,
                infection_constraints=infection_constraint,
            )
            kmf_dict[constraint_name] = kmf
        plot_survival_analysis_results(title_tag=title_tag, kmf_dict=kmf_dict)


def compute_infection_survival_analysis_results(
    pat_inf: pd.DataFrame,  # patient infection information
    pat_stop: pd.DataFrame,  # patient censoring information
    kid_bl: pd.DataFrame,  # organ baseline information
    infection_constraints: dict[str, list[str]]=None,  # {pat_inf_key: [pat_inf_values]}
) -> None:
    """ Plot the number of patients infected over time, constraining the analayis
        to a subset of the infections defined by infection_constraint
    """
    # Select only a certain type of infections, if required
    if infection_constraints is not None:
        for key, valid_values in infection_constraints.items():
            pat_inf = pat_inf[pat_inf[key].isin(valid_values)]
    
    # Get and merge the relevant data
    infections = pat_inf[["patid", "infdate", "inf"]].copy()
    transplants = kid_bl[["patid", "tpxdate"]].copy()

    # Ensure date columns are relevant and in the datetime format
    pd.set_option("future.no_silent_downcasting", True)
    infections["infdate"] = infections["infdate"].replace(pd.Timestamp("9999-01-01"), pd.NaT)
    infections["infdate"] = infections["infdate"].replace(pd.Timestamp("7777-01-01"), pd.NaT)
    infections["infdate"] = pd.to_datetime(infections["infdate"])
    transplants["tpxdate"] = pd.to_datetime(transplants["tpxdate"])

    # Merge infections with transplants, but keeping transplant patids outside constraint
    merged = pd.merge(transplants, infections, on="patid", how="left")

    # Keeps only "clinically relevant" infections
    merged.loc[merged["inf"] != "Infection", "infdate"] = pd.NaT

    # Utility function to select correct infection events
    def time_to_closest_past_transplant(row):
        """ Compute time of infection to the closest transplant event
        """
        # If the event did not detect any infection
        if row["inf"] != "Infection" or pd.isna(row["infdate"]): return pd.NaT

        # Identify all past transplants for this patient
        past_transplants = transplants[
            (transplants["patid"] == row["patid"]) &\
            (transplants["tpxdate"] <= row["infdate"])
        ]
        if past_transplants.empty: return pd.NaT
        
        # Compute the difference between the infection date and the closest transplant event
        closest_transplant = past_transplants.sort_values(by="tpxdate", ascending=False).iloc[0]["tpxdate"]
        time_diff = row["infdate"] - closest_transplant
        
        return time_diff if time_diff.days >= 0 else pd.NaT  # the latter should never happen
    
    # Compute time from infection to closest past transplant event for each patient
    merged["time_to_inf"] = merged.apply(time_to_closest_past_transplant, axis=1)
    merged = merged.sort_values(by=["patid", "time_to_inf"])
    merged_first_inf = merged.drop_duplicates(subset=["patid"], keep="first")

    # Extract a censoring date for all patients of the "pat_stop" dataframe (dropout + dead patients)
    pat_stop.loc[pat_stop["exit"] == "Death", "censdate"] = pat_stop["deathdate"]
    pat_stop.loc[pat_stop["exit"] == "Drop out-loss to follow up", "censdate"] = pat_stop["dropoutdate"]

    # Fill-in a censoring time_to_inf value for all patients without a clinically relevant infection
    merged_first_inf = pd.merge(merged_first_inf, pat_stop[["patid", "censdate"]], on="patid", how="left")
    censored_time_to_inf = merged_first_inf["censdate"] - merged_first_inf["tpxdate"]  # using death or dropout time
    default_time_to_inf = pd.to_datetime("2022-12-31") - merged_first_inf["tpxdate"]  # last possible assesment time
    merged_first_inf["censored_time_to_inf"] = censored_time_to_inf.where(~censored_time_to_inf.isna(), default_time_to_inf)
    merged_first_inf["time_to_inf"] = merged_first_inf["time_to_inf"].fillna(merged_first_inf["censored_time_to_inf"])
    
    # Fit Kaplan-Meier survival curve
    time_to_first_inf = merged_first_inf["time_to_inf"].dt.days
    inf_observed = merged_first_inf["inf"] == "Infection"
    kmf = KaplanMeierFitter()
    kmf.fit(time_to_first_inf, event_observed=inf_observed)

    return kmf


def plot_survival_analysis_results(
    title_tag: str,
    kmf_dict: dict[int, KaplanMeierFitter],
):
    # Initialize the survival curve plot
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Time to first infection (Kaplan-Meier estimate)\n(only for those occuring after transplantation)")
    ax.set_ylim([0.0, 1.0])

    # Plot survival analysis plots, using infection data with different stratifications
    for constraint_name, kmf in kmf_dict.items():
        kmf.plot_survival_function(ax=ax, label=constraint_name)

    # Save polished figure
    ax.set_ylabel("Probability of remaining infection-free")
    ax.set_xlabel("Time after transplantation [days]")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    ax.set_xlim([0, 365])
    plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_survival.png"))
    ax.set_xlim([0, 50])
    plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_survival_zommed.png"))
    plt.close()

    # Initialize the cumulative density plot
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Cumulative incidence of first infection (Kaplan-Meier estimate)\n(only for those occuring after transplantation)")
    ax.set_ylim([0.0, 1.0])

    # Plot cumulative density plots, using infection data with different stratifications
    for constraint_name, kmf in kmf_dict.items():
        kmf.plot_cumulative_density(ax=ax, label=constraint_name)

    # Save polished figure
    ax.set_ylabel("Cumulative probability of infection")
    ax.set_xlabel("Time after transplantation [days]")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    ax.set_xlim([0, 365])
    plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_incidence.png"))
    ax.set_xlim([0, 50])
    plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_incidence_zoomed.png"))
    plt.close()

    # Initialize the probability mass at each even time
    _, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel("Time after transplantation [days]")
    ax.set_ylabel("Probability mass of first infection (smoothed)")
    ax.set_title("Probability mass of first infection (Kaplan-Meier estimate)\n(only for those occuring after transplantation)")

    # Plot probability distribution ("mass") plots, using infection data with different stratifications
    for constraint_name, kmf in kmf_dict.items():
        if constraint_name == "all patients"\
        and kmf_dict.keys() != {"all patients"}:
                continue
        
        cumulative_density_data = kmf.cumulative_density_
        jump_sizes = cumulative_density_data.diff()
        event_point_masses = jump_sizes[jump_sizes != 0].dropna()
        x = event_point_masses.index
        y = event_point_masses.values.squeeze()

        y_new = pd.Series(y, index=x)
        y_mean = y_new.rolling(window=10, center=True).mean()
        y_std = y_new.rolling(window=10, center=True).std()

        ax.plot(x, y_mean, label=constraint_name)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    
    # Save polished figure
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 365])
    plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_mass.png"))
    ax.set_xlim([0, 100])
    plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_mass_zoomed.png"))
    plt.close()


def generate_sex_distribution_plot(
    pat_bl: pd.DataFrame,
    bin_size: int=10,  # in years
) -> None:
    """ Plot the sex distribution of donors and receivers
    """
    # Extract receiver data and calculate age
    pat_bl = pat_bl.groupby("patid").first().reset_index()
    pat_bl = pat_bl[["birthday", "sex"]]
    pat_bl["age"] = (datetime.datetime(2025, 1, 1) - pat_bl["birthday"]).dt.days / 365.25

    # Define age bins and count sexes for all groups
    min_age = int(pat_bl["age"].min()) if not pat_bl.empty else 0
    max_age = int(pat_bl["age"].max()) if not pat_bl.empty else bin_size
    bins = range(min_age, max_age + bin_size, bin_size)
    labels = [f"{i}-{i + bin_size - 1}" for i in bins[:-1]]
    pat_bl["age_group"] = pd.cut(pat_bl["age"], bins=bins, labels=labels, right=False)
    sex_groups = pat_bl.groupby(["age_group", "sex"], observed=True)
    sex_distribution = sex_groups.size().unstack(fill_value=0)
    male_counts = sex_distribution.get("Male", [0] * len(labels))
    female_counts = sex_distribution.get("Female", [0] * len(labels))

    # Plot the histograms for each age category
    _, ax = plt.subplots(figsize=(6, 5))
    bar_width = 0.35
    x = range(len(labels))
    ax.bar(
        x=[i - bar_width/2 for i in x], height=male_counts, width=bar_width,
        label="Male", color="skyblue", edgecolor="black",
    )
    ax.bar(
        x=[i + bar_width/2 for i in x], height=female_counts, width=bar_width,
        label="Female", color="salmon", edgecolor="black",
    )

    # Save polished figure
    ax.set_ylabel("Count")
    ax.set_xlabel(f"Age bin")
    ax.set_title('Sex distribution of donors across age groups')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"sex_distribution.png"))
    plt.close()


def generate_age_distribution_plot(
    patient_bl: pd.DataFrame,
    kidney_bl: pd.DataFrame,
    bin_size: int=4,  # in years
) -> None:
    """ Plot the age distribution of donors and receivers on parallel histograms
    """
    # Extract donor and receiver age list
    kidney_bl = kidney_bl.groupby("patid").first().reset_index()
    patient_bl = patient_bl.groupby("patid").first().reset_index()
    donor_bday_df = kidney_bl["donbirthdate"].dropna()
    donor_bday_df = donor_bday_df[donor_bday_df.between(*csts.VALID_DATE_RANGE)]
    donor_age_df = (datetime.datetime(2025, 1, 1) - donor_bday_df)
    receiver_age_df = (datetime.datetime(2025, 1, 1) - patient_bl["birthday"])
    donor_age_list = donor_age_df.map(lambda t: t.days / 365.25).tolist()
    receiver_age_list = receiver_age_df.map(lambda t: t.days / 365.25).tolist()

    # Compute where bins fall in the histogram
    all_ages = donor_age_list + receiver_age_list
    min_age = int(np.floor(min(all_ages) / bin_size)) * bin_size if all_ages else 0
    max_age = int(np.ceil(max(all_ages) / bin_size)) * bin_size if all_ages else bin_size
    bin_edges_receiver = np.arange(min_age, max_age + bin_size, bin_size) + bin_size / 2
    bin_edges_donor = np.arange(min_age, max_age + bin_size, bin_size)

    # Plot shifted histograms
    _, ax = plt.subplots(figsize=(6, 5))
    shift = bin_size / 4
    ax.hist(
        [age + shift for age in receiver_age_list], bins=bin_edges_receiver,
        alpha=0.75, rwidth=0.5, color="tab:red", edgecolor="black",
        align="mid", label="Receiver",
    )
    ax.hist(
        x=[age - shift for age in donor_age_list], bins=bin_edges_donor,
        alpha=0.75, rwidth=0.5, color="tab:blue", edgecolor="black",
        align="mid", label="Donor",
    )

    # Save polished figure
    ax.set_title(f"Age distribution of donors and receivers")
    ax.set_xlabel("Age [years]")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, max_age + 10, 10))
    ax.grid(axis="y", alpha=0.75)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"age_distribution.png"))
    plt.close()


def generate_infection_type_plots(
    pat_inf: pd.DataFrame,
) -> None:
    """ Plot the total count of infection events by type (bacteria, virus, fungi, parasite)
    """
    # Plot counts for all stratifications of the survival plot dict
    survival_plot_dict = get_survival_plot_dict(pat_inf)
    for title_tag, infection_constraint_dict in survival_plot_dict.items():
        if title_tag not in ["INFTYPE", "INFSITE", "VIRUS", "BACT"]: continue

        # Calculate total event counts for each type, excluding 'Not applicable'
        event_counts = {}
        for type_name, col_dict in infection_constraint_dict.items():
            if col_dict is None: continue
            col_name = next(iter(col_dict.keys()))
            col_valid_values = next(iter(col_dict.values()))
            counts = pat_inf[col_name].value_counts()
            total_events = sum([counts.get(v, 0) for v in col_valid_values])
            event_counts[type_name] = total_events

        # Prepare data for plotting
        types = event_counts.keys()
        counts = event_counts.values()

        # Create the bar plot with label counts
        _, ax = plt.subplots(figsize=(6, 5))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        bars = ax.bar(types, counts, color=colors)
        for bar in bars:
            x_val = bar.get_x() + bar.get_width() / 2
            y_val = bar.get_height()
            ax.text(x_val, y_val, int(y_val), va="bottom", ha="center")

        # Save polished figure
        ax.set_title("Infection event count by type")
        ax.set_xlabel("Infection type")
        ax.set_ylabel("Total event count")
        ax.grid(axis="y", alpha=0.75)
        if title_tag in ["BACT"]:
            for i, label in enumerate(ax.get_xticklabels()):
                if i % 2 != 0:
                    current_pos = label.get_position()
                    label.set_position((current_pos[0], current_pos[1] - 0.05))
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{title_tag}_bar_plot.png"))
        plt.close()


def generate_infection_test_plot(
    pat_inf: pd.DataFrame,
) -> None:
    """ Plot count of Infection vs [No infection or unknown (including NaNs)]
    """
    # Pre-defined counts
    num_inf = pat_inf["inf"].value_counts()["Infection"]
    num_no_inf = pat_inf["inf"].value_counts().sum() - num_inf
    event_counts = {"Infection": num_inf , "No infection\nor unknown": num_no_inf}

    # Prepare data for plotting
    labels = list(event_counts.keys())
    counts = list(event_counts.values())

    # Create bar plot
    _, ax = plt.subplots(figsize=(2.25, 2.5))
    colors = ["tab:red", "tab:gray"]
    bars = ax.bar(labels, counts, width=0.75, color=colors)

    # Add value labels
    for bar in bars:
        x_val = bar.get_x() + bar.get_width() / 2
        y_val = bar.get_height()
        ax.text(x_val, y_val, int(y_val), va="bottom", ha="center")

    # Save polished plot
    ax.set_title("Infection status\nfor all samplings")
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0, top=max(counts) * 1.25)
    ax.set_yticks([])
    ax.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"inf_vs_no_inf_bar_plot.png"))
    plt.close()