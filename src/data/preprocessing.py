# Code adapted from

import re
import sys

import icdmappings
import numpy as np
from pandas import DataFrame, Series, concat

from .util import dataframe_from_csv

###############################
# Non-time series preprocessing
###############################


def transform_gender(gender_series):
    g_map = {"F": 1, "M": 2, "OTHER": 3, "": 0}
    return {
        "gender": gender_series.fillna("").apply(
            lambda s: g_map[s] if s in g_map else g_map["OTHER"]
        )
    }


def transform_marital(marital_series):
    m_map = {"MARRIED": 1, "SINGLE": 2, "WIDOWED": 3, "DIVORCED": 4, "": 0}
    return {
        "marital_status": marital_series.fillna("").apply(
            lambda m: m_map[m] if m in m_map else m_map[""]
        )
    }


def transform_insurance(insurance_series):
    i_map = {"Medicare": 1, "Medicaid": 2, "Other": 3, "": 0}  # TODO: 0 or nan?
    return {
        "insurance": insurance_series.fillna("").apply(
            lambda i: i_map[i] if i in i_map else i_map[""]
        )
    }


def transform_race(race_series):
    r_map = {
        "ASIAN": 1,
        "BLACK": 2,
        "CARIBBEAN ISLAND": 2,
        "HISPANIC": 3,
        "SOUTH AMERICAN": 3,
        "WHITE": 4,
        "MIDDLE EASTERN": 4,
        "PORTUGUESE": 4,
        "AMERICAN INDIAN": 0,
        "NATIVE HAWAIIAN": 0,
        "UNABLE TO OBTAIN": 0,
        "PATIENT DECLINED TO ANSWER": 0,
        "UNKNOWN": 0,
        "OTHER": 0,
        "": 0,
    }

    def aggregate_race(race_str):
        return race_str.replace(" OR ", "/").split(" - ")[0].split("/")[0]

    race_series = race_series.apply(aggregate_race)
    return {
        "race": race_series.fillna("").apply(
            lambda s: r_map[s] if s in r_map else r_map["OTHER"]
        )
    }


def assemble_episodic_data(stays, diagnoses=None):
    data = {
        "hadm_id": stays.hadm_id,  # originally 'stay' was 'stay_id'
        "stay_id": stays.stay_id,
        "age": stays.anchor_age,
        "los": stays.los,
        "mortality": stays.mortality,
        "height": stays.height,
        "weight": stays.weight,
    }
    data.update(transform_gender(stays.gender))
    data.update(transform_race(stays.race))
    data.update(transform_marital(stays.marital_status))
    data.update(transform_insurance(stays.insurance))
    data = DataFrame(data).set_index("hadm_id")
    data = data[
        [
            "race",
            "gender",
            "age",
            "insurance",
            "marital_status",
            "height",
            "weight",
            "los",
            "mortality",
            "stay_id",
        ]
    ]
    # not using diagnoses
    if diagnoses is not None:
        return data.merge(
            extract_diagnosis_labels(diagnoses), left_index=True, right_index=True
        ).set_index("stay_id")
    else:
        return data.set_index("stay_id")


def extract_diagnosis_labels(diagnoses):
    # TODO: Decide on subset of labels to consider, this list is taken from mimic3benchmark repository
    diagnosis_labels = [
        "4019",
        "4280",
        "41401",
        "42731",
        "25000",
        "5849",
        "2724",
        "51881",
        "53081",
        "5990",
        "2720",
        "2859",
        "2449",
        "486",
        "2762",
        "2851",
        "496",
        "V5861",
        "99592",
        "311",
        "0389",
        "5859",
        "5070",
        "40390",
        "3051",
        "412",
        "V4581",
        "2761",
        "41071",
        "2875",
        "4240",
        "V1582",
        "V4582",
        "V5867",
        "4241",
        "40391",
        "78552",
        "5119",
        "42789",
        "32723",
        "49390",
        "9971",
        "2767",
        "2760",
        "2749",
        "4168",
        "5180",
        "45829",
        "4589",
        "73300",
        "5845",
        "78039",
        "5856",
        "4271",
        "4254",
        "4111",
        "V1251",
        "30000",
        "3572",
        "60000",
        "27800",
        "41400",
        "2768",
        "4439",
        "27651",
        "V4501",
        "27652",
        "99811",
        "431",
        "28521",
        "2930",
        "7907",
        "E8798",
        "5789",
        "79902",
        "V4986",
        "V103",
        "42832",
        "E8788",
        "00845",
        "5715",
        "99591",
        "07054",
        "42833",
        "4275",
        "49121",
        "V1046",
        "2948",
        "70703",
        "2809",
        "5712",
        "27801",
        "42732",
        "99812",
        "4139",
        "3004",
        "2639",
        "42822",
        "25060",
        "V1254",
        "42823",
        "28529",
        "E8782",
        "30500",
        "78791",
        "78551",
        "E8889",
        "78820",
        "34590",
        "2800",
        "99859",
        "V667",
        "E8497",
        "79092",
        "5723",
        "3485",
        "5601",
        "25040",
        "570",
        "71590",
        "2869",
        "2763",
        "5770",
        "V5865",
        "99662",
        "28860",
        "36201",
        "56210",
    ]

    diagnoses["value"] = 1

    # convert icd9 labels into icd10
    mapper = icdmappings.Mapper()
    diagnosis_labels = [
        mapper.map(x, source="icd9", target="icd10") for x in diagnosis_labels
    ]

    labels = (
        diagnoses[["hadm_id", "icd_code", "value"]]
        .drop_duplicates()
        .pivot(index="hadm_id", columns="icd_code", values="value")
        .fillna(0)
        .astype(int)
    )

    # if not seen at all then set value to 0 for all stays

    # for label in diagnosis_labels:
    #     if label not in labels:
    #         labels[label] = 0

    missing_labels = DataFrame(
        columns=[
            label for label in diagnosis_labels if label not in labels.icd_code.unique()
        ]
    )
    labels = concat([labels, missing_labels], axis=1)
    return labels.rename(
        dict(
            zip(
                diagnosis_labels,
                ["diagnosis " + d for d in diagnosis_labels],
                strict=False,
            )
        ),
        axis=1,
    )


# phenotyping not used

# def add_hcup_ccs_2015_groups(diagnoses, definitions):
#     def_map = {}
#     for dx in definitions:
#         for code in definitions[dx]['codes']:
#             def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
#     diagnoses['HCUP_CCS_2015'] = diagnoses.ICD9_CODE.apply(lambda c: def_map[c][0] if c in def_map else None)
#     diagnoses['USE_IN_BENCHMARK'] = diagnoses.ICD9_CODE.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
#     return diagnoses


# def make_phenotype_label_matrix(phenotypes, stays=None):
#     phenotypes = phenotypes[['ICUSTAY_ID', 'HCUP_CCS_2015']].loc[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
#     phenotypes['value'] = 1
#     phenotypes = phenotypes.pivot(index='ICUSTAY_ID', columns='HCUP_CCS_2015', values='value')
#     if stays is not None:
#         phenotypes = phenotypes.reindex(stays.ICUSTAY_ID.sort_values())
#     return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)


###################################
# Time series preprocessing
###################################


def read_itemid_to_variable_map(fn, variable_column="LEVEL2"):
    var_map = dataframe_from_csv(fn).fillna("").astype(str)
    # var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != "") & (var_map.COUNT > 0)]
    var_map = var_map[(var_map.STATUS == "ready")]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, "ITEMID", "MIMIC LABEL"]].set_index("ITEMID")
    return var_map.rename(
        {variable_column: "VARIABLE", "MIMIC LABEL": "MIMIC_LABEL"}, axis=1
    )


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on="itemid", right_index=True)


# currently unused
# def read_variable_ranges(fn, variable_column='LEVEL2'):
#     columns = [variable_column, 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH']
#     to_rename = dict(zip(columns, [c.replace(' ', '_') for c in columns]))
#     to_rename[variable_column] = 'VARIABLE'
#     var_ranges = dataframe_from_csv(fn)
#     # var_ranges = var_ranges[variable_column].apply(lambda s: s.lower())
#     var_ranges = var_ranges[columns]
#     var_ranges.rename(to_rename, axis=1, inplace=True)
#     var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
#     var_ranges.set_index('VARIABLE', inplace=True)
#     return var_ranges.loc[var_ranges.notnull().all(axis=1)]

# def remove_outliers_for_variable(events, variable, ranges):
#     if variable not in ranges.index:
#         return events
#     idx = (events.VARIABLE == variable)
#     v = events['value'][idx].copy()
#     v.loc[v < ranges.OUTLIER_LOW[variable]] = np.nan
#     v.loc[v > ranges.OUTLIER_HIGH[variable]] = np.nan
#     v.loc[v < ranges.VALID_LOW[variable]] = ranges.VALID_LOW[variable]
#     v.loc[v > ranges.VALID_HIGH[variable]] = ranges.VALID_HIGH[variable]
#     events.loc[idx, 'value'] = v
#     return events


# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df["value"].astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match("^(\d+)/(\d+)$", s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df["value"].astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match("^(\d+)/(\d+)$", s).group(2))
    return v.astype(float)


# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df['value'] is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df["value"].astype(str)

    v.loc[(df_value_str == "Normal <3 secs") | (df_value_str == "Brisk")] = 0
    v.loc[(df_value_str == "Abnormal >3 secs") | (df_value_str == "Delayed")] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df["value"].astype(float).copy()

    """ The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    """
    # idx = df['valueuom'].fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    """ The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    """
    # idx = df['valueuom'].fillna('').apply(lambda s: 'torr' not in s.lower()) & (df['value'] > 1.0)

    """ The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    """
    is_str = np.array(map(lambda x: x.isinstance(str), list(df["value"])), dtype=bool)
    idx = df["valueuom"].fillna("").apply(lambda s: "torr" not in s.lower()) & (
        is_str | (~is_str & (v > 1.0))
    )

    v.loc[idx] = v[idx] / 100.0
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df["value"].copy()
    idx = v.apply(
        lambda s: isinstance(s, str) and not re.match("^(\d+(\.\d*)?|\.\d+)$", s)
    )
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df["value"].copy()
    idx = v.apply(
        lambda s: isinstance(s, str) and not re.match("^(\d+(\.\d*)?|\.\d+)$", s)
    )
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = v <= 1
    v.loc[idx] = v[idx] * 100.0
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df, min_temp=79):
    # change ""___" to NaN
    v = df["value"].replace(to_replace={"___": np.nan})

    v = v.astype(float).copy()
    idx = (
        df["valueuom"].fillna("").apply(lambda s: "F" in s.lower())
        | df["label"].apply(lambda s: "F" in s.lower())
        | (v >= min_temp)
    )  # noqa: PLR2004
    v.loc[idx] = (v[idx] - 32) * 5.0 / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df["value"].astype(float).copy()
    # ounces
    idx = df["valueuom"].fillna("").apply(lambda s: "oz" in s.lower()) | df[
        "label"
    ].apply(lambda s: "oz" in s.lower())
    v.loc[idx] = v[idx] / 16.0
    # pounds
    idx = (
        idx
        | df["valueuom"].fillna("").apply(lambda s: "lb" in s.lower())
        | df["label"].apply(lambda s: "lb" in s.lower())
    )
    v.loc[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df["value"].astype(float).copy()
    idx = df["valueuom"].fillna("").apply(lambda s: "in" in s.lower()) | df[
        "label"
    ].apply(lambda s: "in" in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure


def clean_events(events):
    clean_fns = {
        # 'Capillary refill rate': clean_crr,
        "Diastolic blood pressure": clean_dbp,
        "Systolic blood pressure": clean_sbp,
        "Fraction inspired oxygen": clean_fio2,
        "Oxygen saturation": clean_o2sat,
        # 'Glucose': clean_lab,
        # 'pH': clean_lab,
        "Temperature": clean_temperature,
        # 'Weight': clean_weight,
        # 'Height': clean_height
    }
    for var_name, clean_fn in clean_fns.items():
        idx = events.label == var_name
        try:
            events.loc[idx, "value"] = clean_fn(events[idx])
        except Exception as e:
            import traceback

            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            sys.exit()
    return events.loc[events["value"].notnull()]
