import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Union, Dict, Any

def cargar_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    print(f"[preproc] Cargando CSV: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"[preproc] CSV cargado. shape={df.shape}")
    return df

def _inferir_target(df: pd.DataFrame, target: Union[str, None]) -> str:
    if target and target in df.columns:
        return target
    if "label" in df.columns:
        return "label"
    return df.columns[-1]

def _imputar(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
    return df

def preparar_matriz(
    df: pd.DataFrame,
    target: Union[str, None] = None,
    categorical_cols: Union[List[str], None] = None,
    encode_categoricals: bool = True,
    scale_numeric: bool = True,
    drop_first_dummy: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    df = df.copy()
    target_col = _inferir_target(df, target)
    print(f"[preproc] Target inferido: '{target_col}'")
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada.")

    df = _imputar(df)
    y_ser = df[target_col]
    X_df = df.drop(columns=[target_col])

    if categorical_cols is not None:
        for col in categorical_cols:
            if col not in X_df.columns:
                raise ValueError(f"Col categórica indicada '{col}' no encontrada.")
        cat_cols = list(categorical_cols)
        num_cols = [c for c in X_df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(X_df[c])]
        forced_cats = [c for c in categorical_cols if pd.api.types.is_numeric_dtype(X_df[c])]
        if forced_cats:
            for c in forced_cats:
                X_df[c] = X_df[c].astype(str)
    else:
        num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        cat_cols = [c for c in X_df.columns if c not in num_cols]

    print(f"[preproc] num_cols={num_cols} | cat_cols={cat_cols}")

    features = []
    parts = []
    metadata: Dict[str, Any] = {"num_means": {}, "num_stds": {}, "cats": {}}

    if num_cols:
        X_num = X_df[num_cols].astype(float).to_numpy()
        if scale_numeric:
            means = np.nanmean(X_num, axis=0)
            stds = np.nanstd(X_num, axis=0)
            stds[stds == 0] = 1.0
            X_num_scaled = (X_num - means) / stds
            parts.append(X_num_scaled)
            for i, col in enumerate(num_cols):
                metadata["num_means"][col] = float(means[i])
                metadata["num_stds"][col] = float(stds[i])
            features.extend(num_cols)
        else:
            parts.append(X_num)
            for i, col in enumerate(num_cols):
                metadata["num_means"][col] = float(np.nanmean(X_num[:, i]))
                metadata["num_stds"][col] = float(np.nanstd(X_num[:, i]))
            features.extend(num_cols)

    if cat_cols:
        if encode_categoricals:
            for col in cat_cols:
                ser = X_df[col].astype(str)
                uniques = sorted(list(pd.Categorical(ser).categories))
                metadata["cats"][col] = uniques
                start_idx = 1 if drop_first_dummy and len(uniques) > 0 else 0
                dummies = []
                dummy_names = []
                for i, val in enumerate(uniques):
                    if i < start_idx:
                        continue
                    col_name = f"{col}__{val}"
                    dummy = (ser.values == val).astype(float).reshape(-1, 1)
                    dummies.append(dummy)
                    dummy_names.append(col_name)
                if dummies:
                    arr = np.hstack(dummies)
                    parts.append(arr)
                    features.extend(dummy_names)
        else:
            X_cat_codes = np.vstack([pd.Categorical(X_df[c].astype(str)).codes for c in cat_cols]).T.astype(float)
            parts.append(X_cat_codes)
            features.extend(cat_cols)

    if parts:
        X = np.hstack(parts)
    else:
        X = np.empty((len(df), 0))

    if pd.api.types.is_numeric_dtype(y_ser):
        y = y_ser.astype(int).values
    else:
        uniques = list(pd.Series(y_ser).unique())
        if len(uniques) == 2:
            mapping = {uniques[0]: 0, uniques[1]: 1}
            y = pd.Series(y_ser).map(mapping).astype(int).values
            metadata["target_mapping"] = mapping
        else:
            raise ValueError(f"Objetivo no binario. Valores únicos: {uniques}")

    print(f"[preproc] Resultado X.shape={X.shape} | y.shape={y.shape} | features_count={len(features)}")
    return X.astype(float), y.astype(int), features, metadata

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42):
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception:
        rng = np.random.RandomState(random_state)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        n_test = int(np.ceil(X.shape[0] * test_size))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def cargar_y_preprocesar_datos(
    path: Union[str, Path],
    target: Union[str, None] = None,
    categorical_cols: Union[List[str], None] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    return_metadata: bool = False,
    encode_categoricals: bool = True,
    scale_numeric: bool = True,
    drop_first_dummy: bool = True
):
    df = cargar_csv(path)
    X, y, features, metadata = preparar_matriz(
        df,
        target=target,
        categorical_cols=categorical_cols,
        encode_categoricals=encode_categoricals,
        scale_numeric=scale_numeric,
        drop_first_dummy=drop_first_dummy
    )
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, random_state=random_state)
    if return_metadata:
        meta_out = {"features": features, **metadata}
        print(f"[preproc] Metadata keys: {list(meta_out.keys())}")
        return X_train, X_test, y_train, y_test, meta_out
    return X_train, X_test, y_train, y_test, features