import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Pedigree Drawer (Family Blocks)", layout="wide")

# =============================
# 默认示例数据（含 spouse_id）
# =============================
DEFAULT_ROWS = [
    {"id":"P4","name":"祖父(父系)","sex":"M","affected":True,  "deceased":True,  "father_id":"","mother_id":"","spouse_id":"P5","proband":False,"birth_order":None},
    {"id":"P5","name":"祖母(父系)","sex":"F","affected":False, "deceased":False, "father_id":"","mother_id":"","spouse_id":"P4","proband":False,"birth_order":None},
    {"id":"P6","name":"外祖父","sex":"M","affected":False, "deceased":False, "father_id":"","mother_id":"","spouse_id":"P7","proband":False,"birth_order":None},
    {"id":"P7","name":"外祖母","sex":"F","affected":True,  "deceased":True,  "father_id":"","mother_id":"","spouse_id":"P6","proband":False,"birth_order":None},

    {"id":"P2","name":"父亲","sex":"M","affected":False, "deceased":False, "father_id":"P4","mother_id":"P5","spouse_id":"P3","proband":False,"birth_order":None},
    {"id":"P3","name":"母亲","sex":"F","affected":False, "deceased":False, "father_id":"P6","mother_id":"P7","spouse_id":"P2","proband":False,"birth_order":None},

    {"id":"P8","name":"姐姐","sex":"F","affected":False, "deceased":False, "father_id":"P2","mother_id":"P3","spouse_id":"P14","proband":False,"birth_order":1},
    {"id":"P1","name":"患者","sex":"F","affected":True,  "deceased":False, "father_id":"P2","mother_id":"P3","spouse_id":"P11","proband":True, "birth_order":2},
    {"id":"P9","name":"弟弟","sex":"M","affected":True,  "deceased":True,  "father_id":"P2","mother_id":"P3","spouse_id":"","proband":False,"birth_order":3},
    {"id":"P10","name":"妹妹","sex":"F","affected":True, "deceased":True,  "father_id":"P2","mother_id":"P3","spouse_id":"P16","proband":False,"birth_order":4},

    {"id":"P11","name":"配偶","sex":"M","affected":False, "deceased":False, "father_id":"","mother_id":"","spouse_id":"P1","proband":False,"birth_order":None},
    {"id":"P14","name":"姐夫","sex":"M","affected":False, "deceased":False, "father_id":"","mother_id":"","spouse_id":"P8","proband":False,"birth_order":None},
    {"id":"P16","name":"妹夫","sex":"M","affected":False, "deceased":False, "father_id":"","mother_id":"","spouse_id":"P10","proband":False,"birth_order":None},

    {"id":"P12","name":"儿子","sex":"M","affected":False, "deceased":False, "father_id":"P11","mother_id":"P1","spouse_id":"","proband":False,"birth_order":1},
    {"id":"P13","name":"女儿","sex":"F","affected":False, "deceased":False, "father_id":"P11","mother_id":"P1","spouse_id":"","proband":False,"birth_order":2},

    {"id":"P15","name":"侄子","sex":"M","affected":False, "deceased":False, "father_id":"P14","mother_id":"P8","spouse_id":"","proband":False,"birth_order":1},
]

# =============================
# 基础工具
# =============================
def to_bool(v):
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in ["true", "1", "yes", "y", "是"]

def to_int_or_none(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def clean_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None

def df_to_people(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        pid = clean_id(r.get("id", ""))
        if not pid:
            continue
        rows.append({
            "id": pid,
            "name": str(r.get("name", "")).strip() or pid,
            "sex": (str(r.get("sex", "U")).strip().upper() or "U"),
            "affected": to_bool(r.get("affected", False)),
            "deceased": to_bool(r.get("deceased", False)),
            "father_id": clean_id(r.get("father_id", "")),
            "mother_id": clean_id(r.get("mother_id", "")),
            "spouse_id": clean_id(r.get("spouse_id", "")),
            "proband": to_bool(r.get("proband", False)),
            "birth_order": to_int_or_none(r.get("birth_order", None)),
        })
    return rows

def get_person_map(people):
    return {p["id"]: p for p in people}

def find_proband_id(people):
    for p in people:
        if p.get("proband"):
            return p["id"]
    return None

def display_person(person_map, pid):
    if pid in person_map:
        p = person_map[pid]
        return f"{p.get('name', pid)}({pid})"
    return str(pid)

# =============================
# 中文亲属称呼解析 + 一键入图（以患者为基准）
# =============================
KINSHIP_ENTRIES = [
    {"canonical":"本人","aliases":["我","自己","本人","患者"],"generation":"同代","sex":"U","paths":[["本人"]]},
    {"canonical":"配偶","aliases":["配偶","爱人","丈夫","妻子","老公","老婆"],"generation":"同代","sex":"U","paths":[["配偶"]]},

    {"canonical":"父亲","aliases":["父亲","爸爸","父","爹"],"generation":"上1代","sex":"M","paths":[["父亲"]]},
    {"canonical":"母亲","aliases":["母亲","妈妈","母","娘"],"generation":"上1代","sex":"F","paths":[["母亲"]]},
    {"canonical":"公公/岳父","aliases":["公公","岳父","丈人"],"generation":"上1代（姻亲）","sex":"M","paths":[["配偶","父亲"]]},
    {"canonical":"婆婆/岳母","aliases":["婆婆","岳母"],"generation":"上1代（姻亲）","sex":"F","paths":[["配偶","母亲"]]},

    {"canonical":"祖父（爷爷）","aliases":["祖父","爷爷"],"generation":"上2代","sex":"M","paths":[["父亲","父亲"]]},
    {"canonical":"祖母（奶奶）","aliases":["祖母","奶奶"],"generation":"上2代","sex":"F","paths":[["父亲","母亲"]]},
    {"canonical":"外祖父（外公）","aliases":["外祖父","外公","姥爷"],"generation":"上2代","sex":"M","paths":[["母亲","父亲"]]},
    {"canonical":"外祖母（外婆）","aliases":["外祖母","外婆","姥姥"],"generation":"上2代","sex":"F","paths":[["母亲","母亲"]]},

    {"canonical":"哥哥","aliases":["哥哥","兄"],"generation":"同代","sex":"M","paths":[["哥哥"]]},
    {"canonical":"弟弟","aliases":["弟弟","弟"],"generation":"同代","sex":"M","paths":[["弟弟"]]},
    {"canonical":"姐姐","aliases":["姐姐","姊姊","姐"],"generation":"同代","sex":"F","paths":[["姐姐"]]},
    {"canonical":"妹妹","aliases":["妹妹","妹"],"generation":"同代","sex":"F","paths":[["妹妹"]]},

    {"canonical":"伯父","aliases":["伯父","伯伯","大伯"],"generation":"上1代（旁系）","sex":"M","paths":[["父亲","哥哥"]]},
    {"canonical":"伯母","aliases":["伯母","大妈"],"generation":"上1代（旁系姻亲）","sex":"F","paths":[["父亲","哥哥","配偶"]]},
    {"canonical":"叔叔","aliases":["叔叔","叔父","小叔"],"generation":"上1代（旁系）","sex":"M","paths":[["父亲","弟弟"]]},
    {"canonical":"婶婶","aliases":["婶婶","婶母","婶子"],"generation":"上1代（旁系姻亲）","sex":"F","paths":[["父亲","弟弟","配偶"]]},
    {"canonical":"姑姑","aliases":["姑姑","姑妈","姑母"],"generation":"上1代（旁系）","sex":"F","paths":[["父亲","姐妹"]]},
    {"canonical":"姑父","aliases":["姑父"],"generation":"上1代（旁系姻亲）","sex":"M","paths":[["父亲","姐妹","配偶"]]},
    {"canonical":"舅舅","aliases":["舅舅","舅父","舅"],"generation":"上1代（旁系）","sex":"M","paths":[["母亲","兄弟"]]},
    {"canonical":"舅妈","aliases":["舅妈","舅母"],"generation":"上1代（旁系姻亲）","sex":"F","paths":[["母亲","兄弟","配偶"]]},
    {"canonical":"姨妈","aliases":["姨妈","阿姨","姨母"],"generation":"上1代（旁系）","sex":"F","paths":[["母亲","姐妹"]]},
    {"canonical":"姨父","aliases":["姨父","姨夫"],"generation":"上1代（旁系姻亲）","sex":"M","paths":[["母亲","姐妹","配偶"]]},

    {"canonical":"儿子","aliases":["儿子"],"generation":"下1代","sex":"M","paths":[["儿子"]]},
    {"canonical":"女儿","aliases":["女儿","闺女"],"generation":"下1代","sex":"F","paths":[["女儿"]]},
    {"canonical":"女婿","aliases":["女婿"],"generation":"下1代（姻亲）","sex":"M","paths":[["女儿","配偶"]]},
    {"canonical":"儿媳","aliases":["儿媳","媳妇"],"generation":"下1代（姻亲）","sex":"F","paths":[["儿子","配偶"]]},

    {"canonical":"侄子","aliases":["侄子"],"generation":"下1代（旁系）","sex":"M","paths":[["兄弟","儿子"]]},
    {"canonical":"侄女","aliases":["侄女"],"generation":"下1代（旁系）","sex":"F","paths":[["兄弟","女儿"]]},
    {"canonical":"外甥","aliases":["外甥"],"generation":"下1代（旁系）","sex":"M","paths":[["姐妹","儿子"]]},
    {"canonical":"外甥女","aliases":["外甥女"],"generation":"下1代（旁系）","sex":"F","paths":[["姐妹","女儿"]]},

    {"canonical":"堂兄弟姐妹","aliases":["堂哥","堂弟","堂姐","堂妹","堂兄","堂兄弟姐妹"],"generation":"同代（旁系）","sex":"U","paths":[["父亲","兄弟","子女"]]},
    {"canonical":"表兄弟姐妹","aliases":["表哥","表弟","表姐","表妹","表兄弟姐妹"],"generation":"同代（旁系）","sex":"U","paths":[["母亲","兄弟姐妹","子女"],["父亲","姐妹","子女"]]},
]


def _normalize_kinship_term(term: str):
    if term is None:
        return ""
    s = str(term).strip().replace(" ", "")
    for token in ["我是", "我", "患者", "先证者", "的", "关系"]:
        s = s.replace(token, "")
    return s


def _path_to_text(path):
    if path == ["本人"]:
        return "患者本人"
    return "患者的" + "的".join(path)


def parse_kinship_term(term: str):
    key = _normalize_kinship_term(term)
    if not key:
        return []
    hits = []
    for entry in KINSHIP_ENTRIES:
        aliases = [_normalize_kinship_term(x) for x in entry["aliases"] + [entry["canonical"]]]
        if key in aliases:
            hits.append(entry)
    return hits


def suggest_kinship_terms(prefix: str, limit=16):
    key = _normalize_kinship_term(prefix)
    if not key:
        return []
    pool = []
    for e in KINSHIP_ENTRIES:
        for a in [e["canonical"], *e["aliases"]]:
            n = _normalize_kinship_term(a)
            if key in n:
                pool.append(a)
    uniq = []
    for x in pool:
        if x not in uniq:
            uniq.append(x)
    return uniq[:limit]


def _find_children_ids(df: pd.DataFrame, parent_id: str, sex: str = None):
    out = normalize_df_columns(df)
    res = []
    for _, row in out.iterrows():
        fid = clean_id(row.get("father_id"))
        mid = clean_id(row.get("mother_id"))
        if parent_id in [fid, mid]:
            pid = clean_id(row.get("id"))
            if not pid:
                continue
            if sex and str(row.get("sex", "U")).upper() != sex:
                continue
            res.append(pid)
    return res


def _find_sibling_ids(df: pd.DataFrame, person_id: str, sex: str = None):
    out = normalize_df_columns(df)
    p = get_person_row_from_df(out, person_id)
    if not p:
        return []
    fid = clean_id(p.get("father_id"))
    mid = clean_id(p.get("mother_id"))
    if not fid and not mid:
        return []
    res = []
    for _, row in out.iterrows():
        pid = clean_id(row.get("id"))
        if not pid or pid == person_id:
            continue
        if clean_id(row.get("father_id")) == fid and clean_id(row.get("mother_id")) == mid:
            if sex and str(row.get("sex", "U")).upper() != sex:
                continue
            res.append(pid)
    return res


def _default_name_for_step(step: str):
    return {
        "父亲": "父亲", "母亲": "母亲", "配偶": "配偶", "哥哥": "哥哥", "弟弟": "弟弟", "姐姐": "姐姐", "妹妹": "妹妹",
        "兄弟": "兄弟", "姐妹": "姐妹", "兄弟姐妹": "兄弟姐妹", "儿子": "儿子", "女儿": "女儿", "子女": "子女"
    }.get(step, step)


def _default_sex_for_step(step: str):
    mapping = {
        "父亲": "M", "母亲": "F", "哥哥": "M", "弟弟": "M", "兄弟": "M", "姐姐": "F", "妹妹": "F", "姐妹": "F",
        "儿子": "M", "女儿": "F"
    }
    return mapping.get(step, "U")


def _ensure_parent(df: pd.DataFrame, child_id: str, role: str):
    out = normalize_df_columns(df)
    child = get_person_row_from_df(out, child_id)
    if not child:
        raise ValueError("人物不存在。")
    col = "father_id" if role == "父亲" else "mother_id"
    current = clean_id(child.get(col))
    if current:
        return out, current
    pname = _default_name_for_step(role)
    psex = _default_sex_for_step(role)
    out, pid = add_person_row(out, name=pname, sex=psex)
    out = set_parent_id_safe(out, child_id, col, pid)
    return out, pid


def _ensure_spouse(df: pd.DataFrame, person_id: str):
    out = normalize_df_columns(df)
    row = get_person_row_from_df(out, person_id)
    if not row:
        raise ValueError("人物不存在。")
    sid = clean_id(row.get("spouse_id"))
    if sid:
        return out, sid
    sex = str(row.get("sex", "U")).upper()
    spouse_sex = "F" if sex == "M" else ("M" if sex == "F" else "U")
    out, sid = add_person_row(out, name="配偶", sex=spouse_sex)
    out = set_spouse_relation_safe(out, person_id, sid)
    return out, sid


def _create_sibling(df: pd.DataFrame, person_id: str, step: str):
    out = normalize_df_columns(df)
    base = get_person_row_from_df(out, person_id)
    if not base:
        raise ValueError("人物不存在。")
    out, sib_id = add_person_row(
        out,
        name=_default_name_for_step(step),
        sex=_default_sex_for_step(step),
        father_id=clean_id(base.get("father_id")) or "",
        mother_id=clean_id(base.get("mother_id")) or ""
    )
    return out, sib_id


def _create_child(df: pd.DataFrame, person_id: str, step: str):
    out = normalize_df_columns(df)
    base = get_person_row_from_df(out, person_id)
    if not base:
        raise ValueError("人物不存在。")
    sex = str(base.get("sex", "U")).upper()
    father_id, mother_id = "", ""
    if sex == "M":
        father_id = person_id
    elif sex == "F":
        mother_id = person_id
    else:
        father_id = person_id
    out, child_id = add_person_row(
        out,
        name=_default_name_for_step(step),
        sex=_default_sex_for_step(step),
        father_id=father_id,
        mother_id=mother_id,
    )
    return out, child_id


def add_person_by_kinship_path(df: pd.DataFrame, proband_id: str, path, target_name: str, target_sex: str, target_affected: bool, target_deceased: bool):
    out = normalize_df_columns(df)
    current_id = proband_id

    if path == ["本人"]:
        out = update_person_fields(out, proband_id, name=target_name, sex=target_sex, affected=target_affected, deceased=target_deceased)
        return out, proband_id, False

    for i, step in enumerate(path):
        is_last = i == len(path) - 1

        if step in ["父亲", "母亲"]:
            out, next_id = _ensure_parent(out, current_id, step)
            current_id = next_id
            if is_last:
                out = update_person_fields(out, current_id, name=target_name, sex=target_sex, affected=target_affected, deceased=target_deceased)
                return out, current_id, False
            continue

        if step == "配偶":
            out, next_id = _ensure_spouse(out, current_id)
            current_id = next_id
            if is_last:
                out = update_person_fields(out, current_id, name=target_name, sex=target_sex, affected=target_affected, deceased=target_deceased)
                return out, current_id, False
            continue

        if step in ["哥哥", "弟弟", "姐姐", "妹妹", "兄弟", "姐妹", "兄弟姐妹"]:
            if is_last:
                out, new_id = _create_sibling(out, current_id, step)
                out = update_person_fields(out, new_id, name=target_name, sex=target_sex, affected=target_affected, deceased=target_deceased)
                return out, new_id, True
            # 中间节点尽量复用，否则创建
            desired_sex = _default_sex_for_step(step)
            cands = _find_sibling_ids(out, current_id, None if desired_sex == "U" else desired_sex)
            if cands:
                current_id = cands[0]
            else:
                out, current_id = _create_sibling(out, current_id, step)
            continue

        if step in ["儿子", "女儿", "子女"]:
            if is_last:
                out, new_id = _create_child(out, current_id, step)
                out = update_person_fields(out, new_id, name=target_name, sex=target_sex, affected=target_affected, deceased=target_deceased)
                return out, new_id, True
            desired_sex = _default_sex_for_step(step)
            cands = _find_children_ids(out, current_id, None if desired_sex == "U" else desired_sex)
            if cands:
                current_id = cands[0]
            else:
                out, current_id = _create_child(out, current_id, step)
            continue

        raise ValueError(f"暂不支持的关系步骤：{step}")

    raise ValueError("关系路径解析失败。")


# =============================
# A2：共同子女 -> 伴侣候选（用户确认）
# =============================
def detect_spouse_candidates_from_children(people):
    person_map = get_person_map(people)
    pair_children = {}

    for child in people:
        fid = child.get("father_id")
        mid = child.get("mother_id")
        cid = child.get("id")
        if not fid or not mid:
            continue
        if fid == mid:
            continue
        if fid not in person_map or mid not in person_map:
            continue
        pair_children.setdefault((fid, mid), []).append(cid)

    candidates = []
    conflicts = []

    for (fid, mid), child_ids in pair_children.items():
        father = person_map[fid]
        mother = person_map[mid]
        f_sp = father.get("spouse_id")
        m_sp = mother.get("spouse_id")

        if f_sp == mid and m_sp == fid:
            status = "already_paired"
            can_apply = False
        elif (not f_sp and not m_sp):
            status = "both_empty"
            can_apply = True
        elif (f_sp == mid and not m_sp):
            status = "fill_mother_side"
            can_apply = True
        elif (m_sp == fid and not f_sp):
            status = "fill_father_side"
            can_apply = True
        else:
            status = "conflict"
            can_apply = False

        item = {
            "pair_key": f"{fid}__{mid}",
            "a": fid,
            "b": mid,
            "a_name": father.get("name", fid),
            "b_name": mother.get("name", mid),
            "a_spouse_id": f_sp,
            "b_spouse_id": m_sp,
            "children": child_ids,
            "status": status,
            "can_apply": can_apply,
        }
        if status == "conflict":
            conflicts.append(item)
        else:
            candidates.append(item)

    return candidates, conflicts

def apply_selected_spouse_candidates_to_df(df: pd.DataFrame, selected_pair_keys):
    if df is None or len(df) == 0 or not selected_pair_keys:
        return df.copy()
    out = df.copy()

    id_to_idx = {}
    for idx, row in out.iterrows():
        pid = clean_id(row.get("id", ""))
        if pid and pid not in id_to_idx:
            id_to_idx[pid] = idx

    for pair_key in selected_pair_keys:
        try:
            a, b = pair_key.split("__", 1)
        except ValueError:
            continue
        if a == b:
            continue
        if a not in id_to_idx or b not in id_to_idx:
            continue

        out.at[id_to_idx[a], "spouse_id"] = b
        out.at[id_to_idx[b], "spouse_id"] = a
    return out

def candidate_status_text(status):
    mapping = {
        "both_empty": "双方未填配偶（可确认）",
        "fill_mother_side": "可补全一侧配偶（可确认）",
        "fill_father_side": "可补全一侧配偶（可确认）",
        "already_paired": "已是配偶（无需处理）",
        "conflict": "与现有配偶信息冲突（需人工处理）",
    }
    return mapping.get(status, status)

# =============================
# B 模式：个人卡片式关系编辑（写回 DataFrame）
# =============================
def normalize_df_columns(df: pd.DataFrame):
    required_cols = [
        "id","name","sex","affected","deceased",
        "father_id","mother_id","spouse_id","proband","birth_order"
    ]
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = None
    return out[required_cols]

def next_person_id(df: pd.DataFrame, prefix="P"):
    df = normalize_df_columns(df)
    used = set()
    for _, row in df.iterrows():
        pid = clean_id(row.get("id"))
        if pid:
            used.add(pid)

    max_n = 0
    for pid in used:
        if pid.startswith(prefix):
            tail = pid[len(prefix):]
            if tail.isdigit():
                max_n = max(max_n, int(tail))

    n = max_n + 1
    while f"{prefix}{n}" in used:
        n += 1
    return f"{prefix}{n}"

def person_options_from_df(df: pd.DataFrame):
    df = normalize_df_columns(df)
    opts = []
    for _, row in df.iterrows():
        pid = clean_id(row.get("id"))
        if not pid:
            continue
        name = str(row.get("name", "")).strip() or pid
        opts.append((f"{name} ({pid})", pid))
    opts.sort(key=lambda x: x[1])
    return opts

def _id_to_row_index(df: pd.DataFrame):
    m = {}
    for idx, row in df.iterrows():
        pid = clean_id(row.get("id"))
        if pid and pid not in m:
            m[pid] = idx
    return m

def add_person_row(
    df: pd.DataFrame,
    name: str,
    sex: str = "U",
    affected: bool = False,
    deceased: bool = False,
    father_id=None,
    mother_id=None,
    spouse_id=None,
    proband: bool = False,
    birth_order=None,
    new_id=None
):
    out = normalize_df_columns(df)

    if new_id is None or not str(new_id).strip():
        new_id = next_person_id(out)
    else:
        new_id = str(new_id).strip()

    existing_ids = {clean_id(v) for v in out["id"].tolist()}
    if new_id in existing_ids:
        raise ValueError(f"id {new_id} 已存在，请换一个。")

    row = {
        "id": new_id,
        "name": (name or new_id).strip(),
        "sex": (sex or "U").upper(),
        "affected": bool(affected),
        "deceased": bool(deceased),
        "father_id": father_id if father_id else "",
        "mother_id": mother_id if mother_id else "",
        "spouse_id": spouse_id if spouse_id else "",
        "proband": bool(proband),
        "birth_order": birth_order if birth_order not in ("", None) else None,
    }
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out, new_id

def add_person_row_with_fixed_id(
    df: pd.DataFrame,
    fixed_id: str,
    name: str,
    sex: str = "U",
    affected: bool = False,
    deceased: bool = False,
    father_id=None,
    mother_id=None,
    spouse_id=None,
    proband: bool = False,
    birth_order=None,
):
    return add_person_row(
        df=df,
        name=name,
        sex=sex,
        affected=affected,
        deceased=deceased,
        father_id=father_id,
        mother_id=mother_id,
        spouse_id=spouse_id,
        proband=proband,
        birth_order=birth_order,
        new_id=fixed_id
    )

def get_person_row_from_df(df: pd.DataFrame, person_id: str):
    out = normalize_df_columns(df)
    for _, row in out.iterrows():
        pid = clean_id(row.get("id"))
        if pid == person_id:
            return row.to_dict()
    return None

def update_person_fields(
    df: pd.DataFrame,
    person_id: str,
    name=None,
    sex=None,
    affected=None,
    deceased=None,
    proband=None,
    birth_order=None,
):
    out = normalize_df_columns(df)
    id_map = _id_to_row_index(out)
    if person_id not in id_map:
        raise ValueError(f"人物 {person_id} 不存在。")
    idx = id_map[person_id]

    if name is not None:
        out.at[idx, "name"] = str(name).strip()
    if sex is not None:
        out.at[idx, "sex"] = str(sex).strip().upper()
    if affected is not None:
        out.at[idx, "affected"] = bool(affected)
    if deceased is not None:
        out.at[idx, "deceased"] = bool(deceased)

    if proband is not None:
        if bool(proband):
            out["proband"] = False
        out.at[idx, "proband"] = bool(proband)

    if birth_order is not None:
        if str(birth_order).strip() == "":
            out.at[idx, "birth_order"] = None
        else:
            out.at[idx, "birth_order"] = int(str(birth_order).strip())

    return out

def set_spouse_relation_safe(df: pd.DataFrame, a_id: str, b_id: str):
    out = normalize_df_columns(df)
    id_map = _id_to_row_index(out)
    if a_id not in id_map or b_id not in id_map:
        raise ValueError("设置配偶失败：人物不存在。")
    if a_id == b_id:
        raise ValueError("不能把自己设为配偶。")

    ai = id_map[a_id]
    bi = id_map[b_id]

    old_a_sp = clean_id(out.at[ai, "spouse_id"])
    if old_a_sp and old_a_sp in id_map and old_a_sp != b_id:
        out.at[id_map[old_a_sp], "spouse_id"] = ""

    old_b_sp = clean_id(out.at[bi, "spouse_id"])
    if old_b_sp and old_b_sp in id_map and old_b_sp != a_id:
        out.at[id_map[old_b_sp], "spouse_id"] = ""

    out.at[ai, "spouse_id"] = b_id
    out.at[bi, "spouse_id"] = a_id
    return out

def clear_spouse_relation_safe(df: pd.DataFrame, a_id: str):
    out = normalize_df_columns(df)
    id_map = _id_to_row_index(out)
    if a_id not in id_map:
        raise ValueError("人物不存在。")
    ai = id_map[a_id]
    old_sp = clean_id(out.at[ai, "spouse_id"])
    out.at[ai, "spouse_id"] = ""
    if old_sp and old_sp in id_map and clean_id(out.at[id_map[old_sp], "spouse_id"]) == a_id:
        out.at[id_map[old_sp], "spouse_id"] = ""
    return out

def set_parent_id_safe(df: pd.DataFrame, child_id: str, role: str, parent_id):
    out = normalize_df_columns(df)
    id_map = _id_to_row_index(out)
    if child_id not in id_map:
        raise ValueError("人物不存在。")
    if role not in ["father_id", "mother_id"]:
        raise ValueError("role 必须是 father_id 或 mother_id。")

    ci = id_map[child_id]

    if parent_id is None or str(parent_id).strip() == "":
        out.at[ci, role] = ""
        return out

    parent_id = str(parent_id).strip()
    if parent_id == child_id:
        raise ValueError("父母不能选择自己。")
    if parent_id not in id_map:
        raise ValueError(f"{role} 指向的人物不存在：{parent_id}")

    out.at[ci, role] = parent_id
    return out

def infer_parent_slot_for_selected(selected_person_row: dict):
    sex = str(selected_person_row.get("sex", "U")).upper()
    if sex == "M":
        return "father"
    if sex == "F":
        return "mother"
    return None

def add_child_under_selected(
    df: pd.DataFrame,
    selected_id: str,
    child_name: str,
    child_sex: str = "U",
    child_birth_order=None,
    other_parent_id=None,
    child_affected: bool = False,
    child_deceased: bool = False,
):
    out = normalize_df_columns(df)
    selected_row = get_person_row_from_df(out, selected_id)
    if not selected_row:
        raise ValueError("当前选中人物不存在。")

    slot = infer_parent_slot_for_selected(selected_row)
    father_id, mother_id = "", ""

    if slot == "father":
        father_id = selected_id
        if other_parent_id:
            mother_id = other_parent_id
    elif slot == "mother":
        mother_id = selected_id
        if other_parent_id:
            father_id = other_parent_id
    else:
        father_id = selected_id
        if other_parent_id:
            mother_id = other_parent_id

    out, child_id = add_person_row(
        out,
        name=child_name,
        sex=child_sex,
        father_id=father_id,
        mother_id=mother_id,
        birth_order=child_birth_order,
        affected=child_affected,
        deceased=child_deceased
    )
    return out, child_id

def add_sibling_of_selected(
    df: pd.DataFrame,
    selected_id: str,
    sibling_name: str,
    sibling_sex: str = "U",
    sibling_birth_order=None,
    sibling_affected: bool = False,
    sibling_deceased: bool = False,
):
    out = normalize_df_columns(df)
    selected_row = get_person_row_from_df(out, selected_id)
    if not selected_row:
        raise ValueError("当前选中人物不存在。")

    father_id = clean_id(selected_row.get("father_id")) or ""
    mother_id = clean_id(selected_row.get("mother_id")) or ""

    out, sib_id = add_person_row(
        out,
        name=sibling_name,
        sex=sibling_sex,
        father_id=father_id,
        mother_id=mother_id,
        birth_order=sibling_birth_order,
        affected=sibling_affected,
        deceased=sibling_deceased
    )
    return out, sib_id

def add_parent_for_selected(
    df: pd.DataFrame,
    selected_id: str,
    parent_role: str,
    parent_name: str,
    parent_sex: str = None,
    parent_affected: bool = False,
    parent_deceased: bool = False,
):
    out = normalize_df_columns(df)
    selected_row = get_person_row_from_df(out, selected_id)
    if not selected_row:
        raise ValueError("当前选中人物不存在。")

    if parent_role not in ["father", "mother"]:
        raise ValueError("parent_role 必须是 father 或 mother。")

    if parent_sex is None:
        parent_sex = "M" if parent_role == "father" else "F"

    out, parent_id = add_person_row(
        out,
        name=parent_name,
        sex=parent_sex,
        affected=parent_affected,
        deceased=parent_deceased
    )
    id_map = _id_to_row_index(out)
    si = id_map[selected_id]
    out.at[si, "father_id" if parent_role == "father" else "mother_id"] = parent_id
    return out, parent_id

# =============================
# 校验 / 布局 / SVG（输出样式不变）
# =============================
def validate_people(people):
    ids = [p.get("id") for p in people]
    if any(not i for i in ids):
        raise ValueError("每个人都必须有 id。")
    if len(ids) != len(set(ids)):
        raise ValueError("存在重复 id（id 不能重复）。")

    person_map = get_person_map(people)
    id_set = set(person_map.keys())

    for p in people:
        if p["sex"] not in ["M", "F", "U"]:
            raise ValueError(f"{p['id']} 的 sex 必须是 M/F/U。")
        for k in ["father_id", "mother_id", "spouse_id"]:
            v = p.get(k)
            if v and v not in id_set:
                raise ValueError(f"{p['id']} 的 {k}={v} 不存在。")
        if p.get("spouse_id") == p["id"]:
            raise ValueError(f"{p['id']} 的 spouse_id 不能指向自己。")

    for p in people:
        sid = p.get("spouse_id")
        if sid:
            other = person_map[sid]
            if other.get("spouse_id") != p["id"]:
                raise ValueError(f"婚配关系需成对填写：{p['id']} ↔ {sid}")

    probands = [p["id"] for p in people if p.get("proband")]
    if len(probands) > 1:
        raise ValueError(f"只能有一个患者（proband=True），当前有多个：{probands}")

    fam_orders = {}
    for p in people:
        fid, mid, bo = p.get("father_id"), p.get("mother_id"), p.get("birth_order")
        if fid and mid and bo is not None:
            key = (fid, mid)
            fam_orders.setdefault(key, set())
            if bo in fam_orders[key]:
                raise ValueError(f"同一父母({fid},{mid})下出现重复 birth_order={bo}")
            fam_orders[key].add(bo)

def compute_generations(people):
    person_map = get_person_map(people)
    gen = {}

    def get_gen(pid, visiting=None):
        if pid in gen:
            return gen[pid]
        if visiting is None:
            visiting = set()
        if pid in visiting:
            return 0
        visiting.add(pid)

        p = person_map[pid]
        parent_gens = []
        for k in ["father_id", "mother_id"]:
            par = p.get(k)
            if par in person_map:
                parent_gens.append(get_gen(par, visiting))
        g = 0 if not parent_gens else max(parent_gens) + 1
        gen[pid] = g
        visiting.remove(pid)
        return g

    for pid in person_map:
        get_gen(pid)
    return gen

def build_child_families(people):
    person_map = get_person_map(people)

    def child_sort_key(cid):
        bo = person_map[cid].get("birth_order")
        return (bo is None, bo if bo is not None else 999999, cid)

    fams = {}
    for p in people:
        fid, mid = p.get("father_id"), p.get("mother_id")
        if fid and mid:
            fams.setdefault((fid, mid), []).append(p["id"])

    for k in fams:
        fams[k] = sorted(fams[k], key=child_sort_key)
    return fams

def build_spouse_pairs(people):
    seen = set()
    pairs = []
    for p in people:
        a = p["id"]
        b = p.get("spouse_id")
        if not b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs

def person_children_map(child_fams):
    m = {}
    for (fid, mid), _children in child_fams.items():
        m.setdefault(fid, []).append((fid, mid))
        m.setdefault(mid, []).append((fid, mid))
    return m

def build_sibling_blocks(sibling_ids, person_map, child_fams, x_gap=160, spouse_gap=105, block_gap=120):
    p2fams = person_children_map(child_fams)
    blocks = []

    for sid in sibling_ids:
        sp = person_map[sid].get("spouse_id")
        fam_keys = p2fams.get(sid, [])

        preferred = None
        if sp:
            for k in fam_keys:
                if set(k) == set([sid, sp]):
                    preferred = k
                    break
        if preferred is None and fam_keys:
            preferred = fam_keys[0]

        children = child_fams.get(preferred, []) if preferred else []

        couple_w = spouse_gap if sp else 0
        child_w = (len(children) - 1) * x_gap if len(children) >= 2 else 0
        width = max(90, couple_w, child_w) + block_gap

        blocks.append({
            "anchor": sid,
            "spouse": sp,
            "family_key": preferred,
            "children": children,
            "width": width,
        })

    return blocks

def fallback_layout(people, gen, child_fams, x_gap, y_gap, margin_x, margin_y):
    coords = {}
    gen_to_ids = {}
    for p in people:
        gen_to_ids.setdefault(gen[p["id"]], []).append(p["id"])
    for g in sorted(gen_to_ids.keys()):
        y = margin_y + g * y_gap
        x = margin_x
        for pid in sorted(gen_to_ids[g]):
            coords[pid] = (x, y)
            x += x_gap
    max_x = max(x for x, _ in coords.values()) if coords else 1000
    max_y = max(y for _, y in coords.values()) if coords else 700
    return coords, child_fams, int(max_x + 260), int(max_y + 300), gen

def structured_layout(people):
    validate_people(people)
    person_map = get_person_map(people)
    gen = compute_generations(people)
    child_fams = build_child_families(people)

    x_gap = 165
    y_gap = 255
    spouse_gap = 110
    margin_x = 120
    margin_y = 110
    upper_side_gap = 220
    block_gap = 125
    reserve_gap = 240

    coords = {}
    proband_id = find_proband_id(people)

    if not proband_id:
        return fallback_layout(people, gen, child_fams, x_gap, y_gap, margin_x, margin_y)

    proband = person_map[proband_id]
    father_id = proband.get("father_id")
    mother_id = proband.get("mother_id")

    if father_id and mother_id and (father_id, mother_id) in child_fams:
        sibling_ids = child_fams[(father_id, mother_id)][:]
    else:
        sibling_ids = [proband_id]
    if proband_id not in sibling_ids:
        sibling_ids.append(proband_id)

    sibling_ids = sorted(sibling_ids, key=lambda pid: (
        person_map[pid].get("birth_order") is None,
        person_map[pid].get("birth_order") if person_map[pid].get("birth_order") is not None else 999999,
        pid
    ))

    y_gp = margin_y
    y_parents = margin_y + y_gap
    y_sibs = margin_y + 2 * y_gap
    y_desc = margin_y + 3 * y_gap
    cx = margin_x + 760

    def _parent_family_siblings(pid):
        if not pid or pid not in person_map:
            return []
        gf = person_map[pid].get("father_id")
        gm = person_map[pid].get("mother_id")
        if gf and gm and (gf, gm) in child_fams:
            return child_fams[(gf, gm)][:]
        return [pid]

    parent_anchor = father_id if father_id else mother_id
    parent_sibling_ids = _parent_family_siblings(parent_anchor)
    if parent_anchor and parent_anchor not in parent_sibling_ids:
        parent_sibling_ids.append(parent_anchor)

    parent_blocks = build_sibling_blocks(
        parent_sibling_ids,
        person_map,
        child_fams,
        x_gap=x_gap,
        spouse_gap=spouse_gap,
        block_gap=block_gap,
    )

    if parent_blocks:
        parent_total_w = sum(b["width"] for b in parent_blocks)
        parent_start_x = cx - parent_total_w / 2
        parent_cursor = parent_start_x
        for b in parent_blocks:
            sid = b["anchor"]
            sp = b["spouse"]
            block_center = parent_cursor + b["width"] / 2
            if sp:
                coords[sid] = (block_center - spouse_gap / 2, y_parents)
                if sp not in coords:
                    coords[sp] = (block_center + spouse_gap / 2, y_parents)
            else:
                coords[sid] = (block_center, y_parents)
            parent_cursor += b["width"]

    if father_id and father_id not in coords:      
        coords[father_id] = (cx - spouse_gap / 2, y_parents)
    if mother_id and mother_id not in coords:
        coords[mother_id] = (cx + spouse_gap / 2, y_parents)

    if father_id in coords and mother_id in coords:
        sib_center_x = (coords[father_id][0] + coords[mother_id][0]) / 2
    else:
        sib_center_x = cx

    blocks = build_sibling_blocks(sibling_ids, person_map, child_fams, x_gap=x_gap, spouse_gap=spouse_gap, block_gap=block_gap)
    total_w = sum(b["width"] for b in blocks) if blocks else 0
    start_x = sib_center_x - total_w / 2

    cursor = start_x
    for b in blocks:
        sid = b["anchor"]
        sp = b["spouse"]
        block_center = cursor + b["width"] / 2
        if sp:
            coords[sid] = (block_center - spouse_gap / 2, y_sibs)
            coords[sp] = (block_center + spouse_gap / 2, y_sibs)
        else:
            coords[sid] = (block_center, y_sibs)
        cursor += b["width"]

    if father_id and father_id in person_map:
        ff = person_map[father_id].get("father_id")
        fm = person_map[father_id].get("mother_id")
        if ff and fm:
            fx, _ = coords[father_id]
            shift = upper_side_gap * 0.6 if (mother_id and mother_id in coords and person_map[mother_id].get("father_id") and person_map[mother_id].get("mother_id")) else 0
            coords[ff] = (fx - upper_side_gap / 2 - shift, y_gp)
            coords[fm] = (fx + upper_side_gap / 2 - shift, y_gp)

    if mother_id and mother_id in person_map:
        mf = person_map[mother_id].get("father_id")
        mm = person_map[mother_id].get("mother_id")
        if mf and mm:
            mx, _ = coords[mother_id]
            shift = upper_side_gap * 0.6 if (father_id and father_id in coords and person_map[father_id].get("father_id") and person_map[father_id].get("mother_id")) else 0
            coords[mf] = (mx - upper_side_gap / 2 + shift, y_gp)
            coords[mm] = (mx + upper_side_gap / 2 + shift, y_gp)

    for b in blocks:
        fam_key = b["family_key"]
        children = b["children"]
        if not fam_key or not children:
            continue
        fid, mid = fam_key
        if fid not in coords or mid not in coords:
            continue
        center_x = (coords[fid][0] + coords[mid][0]) / 2
        n = len(children)
        start_x_children = center_x - ((n - 1) * x_gap) / 2
        for i, cid in enumerate(children):
            coords[cid] = (start_x_children + i * x_gap, y_desc)

    changed = True
    loops = 0
    while changed and loops < 3:
        changed = False
        loops += 1
        for p in people:
            a = p["id"]
            b = p.get("spouse_id")
            if not b:
                continue
            if a in coords and b not in coords:
                ax, ay = coords[a]
                tx = ax + spouse_gap
                if any(abs(ox - tx) < 75 and abs(oy - ay) < 8 for pid2, (ox, oy) in coords.items() if pid2 != a):
                    tx = ax - spouse_gap
                coords[b] = (tx, ay)
                changed = True

    unplaced = [p["id"] for p in people if p["id"] not in coords]
    if unplaced:
        reserve_x = (max(x for x, _ in coords.values()) + reserve_gap) if coords else 1200
        by_gen = {}
        for pid in unplaced:
            by_gen.setdefault(gen.get(pid, 0), []).append(pid)
        for g in sorted(by_gen.keys()):
            y = margin_y + g * y_gap
            x = reserve_x
            for pid in sorted(by_gen[g]):
                coords[pid] = (x, y)
                x += x_gap

    xs = [x for x, _ in coords.values()] if coords else [0]
    min_x, max_x = min(xs), max(xs)
    if min_x < 60:
        shift = 70 - min_x
        for pid in list(coords.keys()):
            x, y = coords[pid]
            coords[pid] = (x + shift, y)
        max_x += shift

    width = int(max_x + 280)
    height = int(max(y for _, y in coords.values()) + 330) if coords else 1000
    return coords, child_fams, width, height, gen

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def line(x1, y1, x2, y2, w=2.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{w}" />'

def choose_arrow_anchor(x, y, width, height, used=None):
    if used is None:
        used = []
    candidates = [
        (x - 105, y - 72, x - 28, y - 20),
        (x + 105, y - 72, x + 28, y - 20),
        (x - 105, y + 72, x - 28, y + 20),
        (x + 105, y + 72, x + 28, y + 20),
    ]
    def score(c):
        tx1, ty1, tx2, ty2 = c
        penalty = 0
        if tx1 < 10 or tx1 > width - 10 or ty1 < 45 or ty1 > height - 10:
            penalty += 1000
        for ex, ey in used:
            if (tx1-ex)**2 + (ty1-ey)**2 < 95**2:
                penalty += 250
        if ty1 > y:
            penalty += 20
        return penalty
    return min(candidates, key=score)

def compute_label_positions(people, coords, base_offset=58):
    rows = {}
    for p in people:
        pid = p["id"]
        if pid not in coords:
            continue
        x, y = coords[pid]
        row_key = round(y)
        rows.setdefault(row_key, []).append((pid, x, y))

    label_pos = {}
    for row_key, items in rows.items():
        row_label_y = row_key + base_offset
        items.sort(key=lambda t: t[1])
        for pid, x, _y in items:
            label_pos[pid] = (x, row_label_y)
    return label_pos

def pedigree_to_svg(people, title="Pedigree", show_labels=True):
    validate_people(people)
    coords, child_fams, width, height, _gen = structured_layout(people)

    r = 26
    base_stroke = 2.6
    proband_stroke = 3.8
    spouse_line_w = 2.4
    label_font = 12
    label_offset = 58
    child_bar_drop = 68

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:white">'
    )
    svg.append(
        f'<text x="{width/2}" y="42" text-anchor="middle" font-size="22" '
        f'font-family="Arial, Microsoft YaHei">{esc(title)}</text>'
    )

    for a, b in build_spouse_pairs(people):
        if a not in coords or b not in coords:
            continue
        ax, ay = coords[a]
        bx, _ = coords[b]
        left_x, right_x = sorted([ax, bx])
        svg.append(line(left_x, ay, right_x, ay, spouse_line_w))

    for (fid, mid), children in child_fams.items():
        if fid not in coords or mid not in coords or not children:
            continue

        fx, fy = coords[fid]
        mx, _ = coords[mid]
        spouse_y = fy
        cx = (fx + mx) / 2
        y_sib = spouse_y + child_bar_drop

        svg.append(line(cx, spouse_y, cx, y_sib, spouse_line_w))

        valid_children = [cid for cid in children if cid in coords]
        if not valid_children:
            continue

        child_points = [(coords[cid][0], coords[cid][1]) for cid in valid_children]
        xs = sorted([x for x, _ in child_points])

        if len(xs) == 1:
            svg.append(line(cx, y_sib, xs[0], y_sib, spouse_line_w))
        else:
            svg.append(line(xs[0], y_sib, xs[-1], y_sib, spouse_line_w))

        for px, py in child_points:
            svg.append(line(px, y_sib, px, py - r, spouse_line_w))

    label_positions = compute_label_positions(people, coords, base_offset=label_offset)

    probands = []
    for p in people:
        pid = p["id"]
        if pid not in coords:
            continue
        x, y = coords[pid]
        sex = p.get("sex", "U")
        affected = bool(p.get("affected", False))
        deceased = bool(p.get("deceased", False))
        proband = bool(p.get("proband", False))
        if proband:
            probands.append(pid)

        fill = "black" if affected else "white"
        stroke_w = proband_stroke if proband else base_stroke

        if sex == "M":
            svg.append(f'<rect x="{x-r}" y="{y-r}" width="{2*r}" height="{2*r}" fill="{fill}" stroke="black" stroke-width="{stroke_w}" />')
        elif sex == "F":
            svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="black" stroke-width="{stroke_w}" />')
        else:
            pts = f"{x},{y-r} {x+r},{y} {x},{y+r} {x-r},{y}"
            svg.append(f'<polygon points="{pts}" fill="{fill}" stroke="black" stroke-width="{stroke_w}" />')

        if deceased:
            ex = r + 12
            ey = r + 12
            svg.append(line(x - ex, y + ey, x + ex, y - ey, 3.0))

    if show_labels:
        for p in people:
            pid = p["id"]
            if pid not in coords or pid not in label_positions:
                continue
            lx, ly = label_positions[pid]
            svg.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="{label_font}" font-family="Arial, Microsoft YaHei">{esc(p.get("name", pid))}</text>')

    used_arrow_tails = []
    for pid in probands:
        x, y = coords[pid]
        ax1, ay1, ax2, ay2 = choose_arrow_anchor(x, y, width, height, used_arrow_tails)
        used_arrow_tails.append((ax1, ay1))
        svg.append(line(ax1, ay1, ax2, ay2, 2.4))

        dx = ax2 - ax1
        dy = ay2 - ay1
        if dx < 0 and dy < 0:
            svg.append(line(ax2, ay2, ax2 + 9, ay2 + 2, 2.4))
            svg.append(line(ax2, ay2, ax2 + 2, ay2 + 9, 2.4))
        elif dx > 0 and dy < 0:
            svg.append(line(ax2, ay2, ax2 - 9, ay2 + 2, 2.4))
            svg.append(line(ax2, ay2, ax2 - 2, ay2 + 9, 2.4))
        elif dx < 0 and dy > 0:
            svg.append(line(ax2, ay2, ax2 + 9, ay2 - 2, 2.4))
            svg.append(line(ax2, ay2, ax2 + 2, ay2 - 9, 2.4))
        else:
            svg.append(line(ax2, ay2, ax2 - 9, ay2 - 2, 2.4))
            svg.append(line(ax2, ay2, ax2 - 2, ay2 - 9, 2.4))

    svg.append("</svg>")
    return "".join(svg)

# =============================
# UI
# =============================
st.title("家系图绘制器（网页版｜家庭块布局）")
st.caption("2.3H：默认隐藏底层数据表；需要时再展开“高级模式”。")

if "pedigree_df" not in st.session_state:
    st.session_state.pedigree_df = pd.DataFrame(DEFAULT_ROWS)

if "spouse_candidate_cache" not in st.session_state:
    st.session_state.spouse_candidate_cache = []
if "spouse_conflict_cache" not in st.session_state:
    st.session_state.spouse_conflict_cache = []
if "spouse_candidate_selected" not in st.session_state:
    st.session_state.spouse_candidate_selected = {}
if "selected_person_id" not in st.session_state:
    st.session_state.selected_person_id = None

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    if st.button("加载示例数据"):
        st.session_state.pedigree_df = pd.DataFrame(DEFAULT_ROWS)
        st.session_state.spouse_candidate_cache = []
        st.session_state.spouse_conflict_cache = []
        st.session_state.spouse_candidate_selected = {}
        st.session_state.selected_person_id = None
        st.rerun()

with c2:
    if st.button("清空表格"):
        st.session_state.pedigree_df = pd.DataFrame(columns=[
            "id","name","sex","affected","deceased","father_id","mother_id","spouse_id","proband","birth_order"
        ])
        st.session_state.spouse_candidate_cache = []
        st.session_state.spouse_conflict_cache = []
        st.session_state.spouse_candidate_selected = {}
        st.session_state.selected_person_id = None
        st.rerun()

with c3:
    show_labels = st.checkbox("显示姓名标签", value=True)

with c4:
    use_spouse_candidate_confirm = st.checkbox("共同子女生成配偶候选（需确认）", value=True)

st.markdown("## 中文亲属称呼一键入图（以患者为基准）")
with st.expander("输入称呼并直接创建人物", expanded=False):
    st.caption("输入称呼后，系统会识别该人物与患者关系；你可填写姓名、患病/死亡并直接加入家系图。")

    people_now = df_to_people(normalize_df_columns(st.session_state.pedigree_df))
    proband_id_now = find_proband_id(people_now)
    if not proband_id_now:
        st.warning("当前还没有患者（proband=True）。请先在下方人物编辑区勾选一位患者。")
    else:
        kinship_term = st.text_input("输入亲属称呼", value="", placeholder="例如：婶婶")
        matches = parse_kinship_term(kinship_term) if kinship_term.strip() else []

        if kinship_term.strip() and not matches:
            st.warning("未找到该称呼映射，请尝试：" + "、".join(suggest_kinship_terms(kinship_term) or ["父亲", "母亲", "婶婶", "舅舅"]))

        if matches:
            option_pairs = []
            option_to_meta = {}
            for m in matches:
                for pth in m["paths"]:
                    label = f"{m['canonical']} ｜ {_path_to_text(pth)}"
                    option_pairs.append(label)
                    option_to_meta[label] = (m, pth)

            selected_option = st.selectbox("选择具体关系路径", options=option_pairs, key="kinship_selected_path")
            selected_entry, selected_path = option_to_meta[selected_option]

            st.info(f"将创建：{_path_to_text(selected_path)}")
            k1, k2, k3, k4 = st.columns([2, 1, 1, 1])
            with k1:
                new_person_name = st.text_input("该亲属姓名/称谓", value=selected_entry["canonical"], key="kinship_new_name")
            with k2:
                sex_default = selected_entry.get("sex", "U")
                idx = ["M", "F", "U"].index(sex_default if sex_default in ["M", "F", "U"] else "U")
                new_person_sex = st.selectbox("性别", options=["M", "F", "U"], index=idx, key="kinship_new_sex")
            with k3:
                new_person_affected = st.checkbox("患病", value=False, key="kinship_new_aff")
            with k4:
                new_person_deceased = st.checkbox("死亡", value=False, key="kinship_new_dec")

            if st.button("按该称呼加入家系图", type="primary"):
                try:
                    if not new_person_name.strip():
                        raise ValueError("请填写姓名/称谓。")
                    new_df, new_id, created_new = add_person_by_kinship_path(
                        st.session_state.pedigree_df,
                        proband_id_now,
                        selected_path,
                        target_name=new_person_name.strip(),
                        target_sex=new_person_sex,
                        target_affected=new_person_affected,
                        target_deceased=new_person_deceased,
                    )
                    st.session_state.pedigree_df = normalize_df_columns(new_df)
                    st.session_state.selected_person_id = new_id
                    if created_new:
                        st.success(f"已新增人物：{new_person_name}（{new_id}）。")
                    else:
                        st.success(f"已更新关系位对应人物：{new_person_name}（{new_id}）。")
                    st.rerun()
                except Exception as e:
                    st.error(f"加入失败：{e}")

        with st.popover("查看支持的称呼（完整列表）"):
            grouped = {}
            for e in KINSHIP_ENTRIES:
                grouped.setdefault(e["generation"], []).append(e)
            for g in ["上2代", "上1代", "上1代（旁系）", "上1代（旁系姻亲）", "上1代（姻亲）", "同代", "同代（旁系）", "下1代", "下1代（旁系）", "下1代（姻亲）"]:
                if g in grouped:
                    st.markdown(f"**{g}**")
                    st.write("、".join(sorted({x["canonical"] for x in grouped[g]})))


st.markdown("## B 模式：围绕人物添加/编辑关系（推荐）")

working_df = normalize_df_columns(st.session_state.pedigree_df)
person_opts = person_options_from_df(working_df)

with st.expander("还没有人物？先创建一个起始人物", expanded=(len(person_opts) == 0)):
    with st.form("form_create_root_person", clear_on_submit=True):
        root_name = st.text_input("姓名/称谓", value="")
        root_sex = st.selectbox("性别", options=["M", "F", "U"], index=2, key="root_person_sex")
        root_proband = st.checkbox("设为患者/先证者（proband）", value=False)
        root_affected = st.checkbox("患病(affected)", value=False)
        root_deceased = st.checkbox("死亡(deceased)", value=False)
        create_root_btn = st.form_submit_button("创建起始人物")
        if create_root_btn:
            try:
                if not root_name.strip():
                    raise ValueError("请填写姓名。")
                new_df, new_id = add_person_row(
                    working_df,
                    name=root_name.strip(),
                    sex=root_sex,
                    proband=root_proband,
                    affected=root_affected,
                    deceased=root_deceased
                )
                st.session_state.pedigree_df = new_df
                st.session_state.selected_person_id = new_id
                st.rerun()
            except Exception as e:
                st.error(f"创建失败：{e}")

if not person_opts:
    st.info("当前还没有人物。先创建起始人物。")
else:
    all_ids = [pid for _, pid in person_opts]
    if st.session_state.selected_person_id not in all_ids:
        st.session_state.selected_person_id = all_ids[0]

    label_to_id = {label: pid for label, pid in person_opts}
    id_to_label = {pid: label for label, pid in person_opts}

    selected_label = st.selectbox(
        "选择人物（个人卡片）",
        options=[id_to_label[pid] for pid in all_ids],
        index=all_ids.index(st.session_state.selected_person_id),
        key="person_card_selectbox"
    )
    selected_id = label_to_id[selected_label]
    st.session_state.selected_person_id = selected_id

    # refresh
    working_df = normalize_df_columns(st.session_state.pedigree_df)
    person_opts = person_options_from_df(working_df)
    existing_ids = set([pid for _, pid in person_opts])
    id_to_label = {pid: label for label, pid in person_opts}
    label_list = [lbl for lbl, _ in person_opts]
    lbl_to_id = {lbl: pid for lbl, pid in person_opts}
    id_to_lbl = {pid: lbl for lbl, pid in person_opts}
    selected_row = get_person_row_from_df(working_df, selected_id)

    if selected_row:
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.metric("当前人物", f"{selected_row.get('name','')} ({selected_id})")
        with i2:
            st.metric("性别", str(selected_row.get("sex","U")))
        with i3:
            st.metric("父亲ID", clean_id(selected_row.get("father_id")) or "未填")
        with i4:
            st.metric("母亲ID", clean_id(selected_row.get("mother_id")) or "未填")
        st.caption(f"配偶ID：{clean_id(selected_row.get('spouse_id')) or '未填'}")

        # 编辑当前人物（含关系）
        with st.expander("编辑当前人物（不用底层数据表）", expanded=False):
            blank_label = "（清空/不填）"
            sex_opts = ["M", "F", "U"]
            cur_sex = str(selected_row.get("sex","U")).upper()
            cur_bo = selected_row.get("birth_order", None)

            # spouse options exclude self
            spouse_options = [blank_label] + [lbl for lbl, pid in person_opts if pid != selected_id]

            cur_f = clean_id(selected_row.get("father_id"))
            cur_m = clean_id(selected_row.get("mother_id"))
            cur_s = clean_id(selected_row.get("spouse_id"))

            def _pick_parent(default_id, key):
                options = [blank_label] + label_list
                if default_id and default_id in id_to_lbl:
                    idx = 1 + label_list.index(id_to_lbl[default_id])
                else:
                    idx = 0
                return st.selectbox("", options=options, index=idx, key=key)

            def _pick_spouse(default_id, key):
                if default_id and default_id in id_to_lbl and default_id != selected_id:
                    lbl = id_to_lbl[default_id]
                    idx = spouse_options.index(lbl) if lbl in spouse_options else 0
                else:
                    idx = 0
                return st.selectbox("", options=spouse_options, index=idx, key=key)

            with st.form(f"edit_person_{selected_id}", clear_on_submit=False):
                new_name = st.text_input("姓名/称谓", value=str(selected_row.get("name","") or ""))
                new_sex = st.selectbox("性别", options=sex_opts, index=sex_opts.index(cur_sex) if cur_sex in sex_opts else 2)
                new_aff = st.checkbox("患病(affected)", value=bool(selected_row.get("affected", False)))
                new_dec = st.checkbox("死亡(deceased)", value=bool(selected_row.get("deceased", False)))
                new_pro = st.checkbox("患者/先证者(proband)", value=bool(selected_row.get("proband", False)))
                new_bo = st.text_input("出生顺序 birth_order（可选，留空=不填）", value="" if cur_bo in (None,"","nan") else str(cur_bo))

                st.markdown("**关系编辑（可清空）**")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.caption("父亲 father_id")
                    father_choice = _pick_parent(cur_f, key=f"edit_father_{selected_id}")
                with r2:
                    st.caption("母亲 mother_id")
                    mother_choice = _pick_parent(cur_m, key=f"edit_mother_{selected_id}")
                with r3:
                    st.caption("配偶 spouse_id")
                    spouse_choice = _pick_spouse(cur_s, key=f"edit_spouse_{selected_id}")

                submit = st.form_submit_button("保存修改")
                if submit:
                    try:
                        df2 = working_df.copy()
                        df2 = update_person_fields(
                            df2,
                            person_id=selected_id,
                            name=new_name,
                            sex=new_sex,
                            affected=new_aff,
                            deceased=new_dec,
                            proband=new_pro,
                            birth_order=new_bo
                        )

                        father_id_new = "" if father_choice == blank_label else lbl_to_id.get(father_choice, "")
                        mother_id_new = "" if mother_choice == blank_label else lbl_to_id.get(mother_choice, "")
                        df2 = set_parent_id_safe(df2, selected_id, "father_id", father_id_new)
                        df2 = set_parent_id_safe(df2, selected_id, "mother_id", mother_id_new)

                        if spouse_choice == blank_label:
                            df2 = clear_spouse_relation_safe(df2, selected_id)
                        else:
                            spouse_id_new = lbl_to_id.get(spouse_choice)
                            if not spouse_id_new:
                                raise ValueError("配偶选择解析失败。")
                            df2 = set_spouse_relation_safe(df2, selected_id, spouse_id_new)

                        st.session_state.pedigree_df = df2
                        st.success("已保存。")
                        st.rerun()
                    except Exception as e:
                        st.error(f"保存失败：{e}")

    

    tab1, tab2, tab3, tab4 = st.tabs(["添加配偶/伴侣", "添加子女", "添加兄弟姐妹", "添加父母"])

    with tab1:
        t1a, t1b = st.columns(2)

        with t1a:
            with st.form("form_add_spouse_new", clear_on_submit=True):
                spouse_name = st.text_input("配偶/伴侣姓名", value="")
                spouse_sex = st.selectbox("配偶性别", options=["M","F","U"], index=2, key="new_spouse_sex")
                spouse_affected = st.checkbox("配偶患病", value=False)
                spouse_deceased = st.checkbox("配偶死亡", value=False)
                submitted = st.form_submit_button("添加新配偶/伴侣")
                if submitted:
                    try:
                        if not spouse_name.strip():
                            raise ValueError("请填写配偶姓名。")
                        new_df, new_spouse_id = add_person_row(
                            working_df,
                            name=spouse_name.strip(),
                            sex=spouse_sex,
                            affected=spouse_affected,
                            deceased=spouse_deceased
                        )
                        new_df = set_spouse_relation_safe(new_df, selected_id, new_spouse_id)
                        st.session_state.pedigree_df = new_df
                        st.rerun()
                    except Exception as e:
                        st.error(f"添加失败：{e}")

        with t1b:
            with st.form("form_bind_existing_spouse", clear_on_submit=False):
                existing_spouse_opts = [x for x in person_opts if x[1] != selected_id]
                if existing_spouse_opts:
                    chosen_label = st.selectbox("选择已有的人物", options=[x[0] for x in existing_spouse_opts], key="bind_existing_spouse_select")
                    chosen_id = dict(existing_spouse_opts)[chosen_label]
                    submitted = st.form_submit_button("设为配偶（自动解绑旧配偶）")
                    if submitted:
                        try:
                            new_df = set_spouse_relation_safe(working_df, selected_id, chosen_id)
                            st.session_state.pedigree_df = new_df
                            st.rerun()
                        except Exception as e:
                            st.error(f"绑定失败：{e}")
                else:
                    st.caption("暂无可选人物（至少需要 2 个人）。")
                    st.form_submit_button("设为配偶", disabled=True)

    with tab2:
        with st.form("form_add_child", clear_on_submit=True):
            child_name = st.text_input("子女姓名", value="")
            child_sex = st.selectbox("子女性别", options=["M","F","U"], index=2, key="new_child_sex")
            child_affected = st.checkbox("子女患病", value=False)
            child_deceased = st.checkbox("子女死亡", value=False)
            bo_raw = st.text_input("出生顺序（可选，填数字如 1/2/3）", value="")

            cur_sp = clean_id(selected_row.get("spouse_id")) if selected_row else None
            other_parent_options = [("（留空）", "")]
            if cur_sp and cur_sp in id_to_label:
                other_parent_options.append((f"当前配偶优先：{id_to_label[cur_sp]}", cur_sp))
            for label, pid in person_opts:
                if pid != selected_id and pid != cur_sp:
                    other_parent_options.append((label, pid))

            other_label = st.selectbox("另一个家长（可选）", options=[x[0] for x in other_parent_options], index=1 if cur_sp else 0, key="other_parent_for_child")
            other_id = dict(other_parent_options).get(other_label, "")

            submitted = st.form_submit_button("添加子女")
            if submitted:
                try:
                    if not child_name.strip():
                        raise ValueError("请填写子女姓名。")
                    if other_id == selected_id:
                        raise ValueError("另一个家长不能选择当前人物自己。")
                    bo = int(bo_raw.strip()) if str(bo_raw).strip() else None

                    new_df, _ = add_child_under_selected(
                        working_df,
                        selected_id=selected_id,
                        child_name=child_name.strip(),
                        child_sex=child_sex,
                        child_birth_order=bo,
                        other_parent_id=other_id or None,
                        child_affected=child_affected,
                        child_deceased=child_deceased
                    )
                    st.session_state.pedigree_df = new_df
                    st.rerun()
                except Exception as e:
                    st.error(f"添加失败：{e}")

    with tab3:
        with st.form("form_add_sibling", clear_on_submit=True):
            sib_name = st.text_input("兄弟姐妹姓名", value="")
            sib_sex = st.selectbox("兄弟姐妹性别", options=["M","F","U"], index=2, key="new_sib_sex")
            sib_affected = st.checkbox("兄弟姐妹患病", value=False)
            sib_deceased = st.checkbox("兄弟姐妹死亡", value=False)
            bo_raw = st.text_input("出生顺序（可选，填数字如 1/2/3）", value="", key="sib_birth_order_input")

            submitted = st.form_submit_button("添加兄弟姐妹")
            if submitted:
                try:
                    if not sib_name.strip():
                        raise ValueError("请填写兄弟姐妹姓名。")
                    bo = int(bo_raw.strip()) if str(bo_raw).strip() else None

                    new_df, _ = add_sibling_of_selected(
                        working_df,
                        selected_id=selected_id,
                        sibling_name=sib_name.strip(),
                        sibling_sex=sib_sex,
                        sibling_birth_order=bo,
                        sibling_affected=sib_affected,
                        sibling_deceased=sib_deceased
                    )
                    st.session_state.pedigree_df = new_df
                    st.rerun()
                except Exception as e:
                    st.error(f"添加失败：{e}")

    with tab4:
        left, right = st.columns(2)
        with left:
            with st.form("form_add_father", clear_on_submit=True):
                father_name = st.text_input("父亲姓名", value="")
                father_affected = st.checkbox("父亲患病", value=False)
                father_deceased = st.checkbox("父亲死亡", value=False)
                submitted = st.form_submit_button("添加父亲")
                if submitted:
                    try:
                        if not father_name.strip():
                            raise ValueError("请填写父亲姓名。")
                        new_df, _ = add_parent_for_selected(
                            working_df,
                            selected_id=selected_id,
                            parent_role="father",
                            parent_name=father_name.strip(),
                            parent_sex="M",
                            parent_affected=father_affected,
                            parent_deceased=father_deceased
                        )
                        st.session_state.pedigree_df = new_df
                        st.rerun()
                    except Exception as e:
                        st.error(f"添加失败：{e}")
        with right:
            with st.form("form_add_mother", clear_on_submit=True):
                mother_name = st.text_input("母亲姓名", value="")
                mother_affected = st.checkbox("母亲患病", value=False)
                mother_deceased = st.checkbox("母亲死亡", value=False)
                submitted = st.form_submit_button("添加母亲")
                if submitted:
                    try:
                        if not mother_name.strip():
                            raise ValueError("请填写母亲姓名。")
                        new_df, _ = add_parent_for_selected(
                            working_df,
                            selected_id=selected_id,
                            parent_role="mother",
                            parent_name=mother_name.strip(),
                            parent_sex="F",
                            parent_affected=mother_affected,
                            parent_deceased=mother_deceased
                        )
                        st.session_state.pedigree_df = new_df
                        st.rerun()
                    except Exception as e:
                        st.error(f"添加失败：{e}")

# =============================
# 高级模式：隐藏的底层数据表 + A2（默认不显示）
# =============================
with st.expander("高级模式（底层数据表 / 配偶候选 / 批量修改）", expanded=False):
    st.caption("正常使用不用管这里。只有当你要批量改 id 或排查数据时再展开。")

    edited_df = st.data_editor(
        st.session_state.pedigree_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("id", help="唯一编号，例如 P1/P2"),
            "name": st.column_config.TextColumn("姓名/称谓"),
            "sex": st.column_config.SelectboxColumn("性别", options=["M", "F", "U"]),
            "affected": st.column_config.CheckboxColumn("患病"),
            "deceased": st.column_config.CheckboxColumn("死亡"),
            "father_id": st.column_config.TextColumn("父亲id"),
            "mother_id": st.column_config.TextColumn("母亲id"),
            "spouse_id": st.column_config.TextColumn("配偶id"),
            "proband": st.column_config.CheckboxColumn("患者(先证者)"),
            "birth_order": st.column_config.NumberColumn("出生顺序", step=1, min_value=1),
        },
        key="data_editor_pedigree_hidden"
    )

    if st.button("保存底层表格修改（应用到系统）", type="secondary"):
        st.session_state.pedigree_df = normalize_df_columns(edited_df)
        st.success("已保存。")
        st.rerun()

    st.markdown("#### 配偶候选（A2，可选辅助）")
    scan_c1, scan_c2 = st.columns([1, 3])
    with scan_c1:
        if st.button("扫描配偶候选", key="scan_candidates_btn"):
            try:
                people_tmp = df_to_people(normalize_df_columns(edited_df))
                candidates, conflicts = detect_spouse_candidates_from_children(people_tmp)
                st.session_state.spouse_candidate_cache = candidates
                st.session_state.spouse_conflict_cache = conflicts
                st.session_state.spouse_candidate_selected = {c["pair_key"]: True for c in candidates if c["can_apply"]}
                st.success("已扫描。")
            except Exception as e:
                st.error(f"扫描失败：{e}")

    with scan_c2:
        st.caption("只填了孩子的 father_id/mother_id，但没填 spouse_id 时，用这个一键回填。")

    candidates = st.session_state.spouse_candidate_cache
    conflicts = st.session_state.spouse_conflict_cache

    if use_spouse_candidate_confirm and (candidates or conflicts):
        person_map_preview = get_person_map(df_to_people(normalize_df_columns(edited_df)))

        if candidates:
            st.markdown("**可确认候选**")
            for c in candidates:
                a_txt = f"{c['a_name']}({c['a']})"
                b_txt = f"{c['b_name']}({c['b']})"
                child_txt = "，".join(display_person(person_map_preview, cid) for cid in c["children"]) or "无"
                status_txt = candidate_status_text(c["status"])
                if c["can_apply"]:
                    current_val = st.session_state.spouse_candidate_selected.get(c["pair_key"], True)
                    checked = st.checkbox(
                        f"{a_txt} ↔ {b_txt} ｜ {status_txt} ｜ 共同子女：{child_txt}",
                        value=current_val,
                        key=f"cand_chk_{c['pair_key']}"
                    )
                    st.session_state.spouse_candidate_selected[c["pair_key"]] = checked
                else:
                    st.caption(f"• {a_txt} ↔ {b_txt} ｜ {status_txt} ｜ 共同子女：{child_txt}")

            if st.button("应用所选候选配偶关系（回填 spouse_id）", type="secondary", key="apply_candidates_btn"):
                try:
                    selected_keys = {k for k, v in st.session_state.spouse_candidate_selected.items() if v}
                    new_df = apply_selected_spouse_candidates_to_df(normalize_df_columns(edited_df), selected_keys)
                    st.session_state.pedigree_df = normalize_df_columns(new_df)
                    st.success("已回填 spouse_id。")
                    st.rerun()
                except Exception as e:
                    st.error(f"应用失败：{e}")

        if conflicts:
            st.markdown("**冲突候选（需人工处理）**")
            for c in conflicts:
                a_txt = f"{c['a_name']}({c['a']})"
                b_txt = f"{c['b_name']}({c['b']})"
                a_sp = display_person(person_map_preview, c["a_spouse_id"]) if c["a_spouse_id"] else "空"
                b_sp = display_person(person_map_preview, c["b_spouse_id"]) if c["b_spouse_id"] else "空"
                child_txt = "，".join(display_person(person_map_preview, cid) for cid in c["children"])
                st.warning(
                    f"{a_txt} & {b_txt}（共同子女：{child_txt}）与现有 spouse_id 冲突："
                    f"{a_txt} 当前配偶={a_sp}；{b_txt} 当前配偶={b_sp}"
                )

# =============================
# 生成图
# =============================
graph_title = st.text_input("图标题", value="Pedigree")
if st.button("生成家系图", type="primary"):
    try:
        people = df_to_people(normalize_df_columns(st.session_state.pedigree_df))
        if len(people) == 0:
            st.warning("当前没有人物。")
        else:
            svg_html = pedigree_to_svg(people, title=graph_title, show_labels=show_labels)
            st.markdown("### 生成结果")
            components.html(svg_html, height=1100, scrolling=True)
    except Exception as e:
        st.error(f"生成失败：{e}")
