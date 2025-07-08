import pandas as pd
from bertopic import BERTopic
import src.constants as constants
csts = constants.ConstantsNamespace()


def main():
    """ Test script to categorize semi-free text
    """
    # Load patient patient infection data sheet
    data_dict = pd.read_pickle(csts.PICKLE_DATA_PATH)
    pat_inf = data_dict[csts.PATIENT_INFECTION_SHEET]
    
    # Select infection site - other (a "semi-free" text variable) and normalize it
    docs = pat_inf["infsiteother_0"].copy().dropna().tolist()
    docs = [d for d in docs if d != "Not applicable"]
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    # TODO: FOR EACH CLUSTER, TAKE A GOOD REPRESENTATIVE,
    # PAR EXEMPLE LE SAMPLE QUI APPARAIT LE + SOUVENT DANS LE CLUSTER
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
