// The config dialog: the blueprint's recommended settings, its modules' args and dimos's global config, each saved
// for every launch of this blueprint (PUT /dimos/blueprints/{name}/config, PUT /dimos/global-config).
import { Alert, Button, Modal, Segmented, Select, TextInput, Toggle } from "dimos-react";
import { type ReactNode, useEffect, useState } from "react";
import { errorText, getJson, send } from "./api.ts";
import {
  asText,
  type BlueprintConfig,
  type FieldSpec,
  type GlobalConfig,
  globalFields,
  HIDDEN,
  moduleFields,
  parseText,
  same,
  valueText,
} from "./config_fields.ts";

type Overrides = Record<string, unknown>;
type ModuleOverrides = Record<string, Record<string, unknown>>;

export interface Saved {
  global: Overrides;
  modules: ModuleOverrides;
  defaults: Overrides;
}

interface Recommended {
  id: string;
  kind?: string;
  key?: string;
  scope?: string;
  label: string;
  placeholder?: string;
  docs?: string;
  when?: Record<string, unknown>;
  choices?: { label: string; value?: unknown; set?: Record<string, unknown> }[];
}

async function recommendedOf(name: string): Promise<Recommended[]> {
  try {
    const { robots } = await getJson<
      {
        robots?: Record<
          string,
          { blueprints?: Record<string, { recommended_config?: Recommended[] }> }
        >;
      }
    >("robots");
    for (const robot of Object.values(robots ?? {})) {
      const entry = robot.blueprints?.[name];
      if (entry) {
        return entry.recommended_config ?? [];
      }
    }
  } catch {
    // no robots: no recommended settings
  }
  return [];
}

function withModule(overrides: ModuleOverrides, spec: FieldSpec, value: unknown): ModuleOverrides {
  const next: ModuleOverrides = JSON.parse(JSON.stringify(overrides));
  const module = spec.module!;
  next[module] ??= {};
  if (value === undefined) {
    delete next[module][spec.name];
  } else {
    next[module][spec.name] = value;
  }
  if (Object.keys(next[module]).length === 0) {
    delete next[module];
  }
  return next;
}

function Choose(
  { labels, current, pick, name }: {
    labels: string[];
    current: number;
    pick: (index: number) => void;
    name: string;
  },
) {
  return labels.length <= 4
    ? (
      <Segmented
        size="sm"
        label={name}
        items={labels.map((label, index) => ({ value: index, label }))}
        value={current < 0 ? null : current}
        onChange={(index) => index !== current && pick(index)}
      />
    )
    : (
      <Select
        size="sm"
        aria-label={name}
        value={current}
        placeholder="—"
        options={labels.map((label, index) => [index, label] as const)}
        onChange={pick}
      />
    );
}

function Reset({ spec, onReset }: { spec: FieldSpec; onReset: () => void }) {
  return (
    <button
      type="button"
      className="reset"
      title={`back to ${valueText(spec.base)}`}
      aria-label={`reset ${spec.name}`}
      onClick={onReset}
    >
      reset
    </button>
  );
}

function Input(
  { spec, value, onSet }: { spec: FieldSpec; value: unknown; onSet: (value: unknown) => void },
) {
  const [bad, setBad] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  if (spec.kind === "bool") {
    return <Toggle size="sm" aria-label={spec.name} value={value === true} onChange={onSet} />;
  }
  if (spec.kind === "enum") {
    const options = [...(spec.nullable ? [null] : []), ...(spec.options ?? [])];
    return (
      <Select
        size="sm"
        aria-label={spec.name}
        value={JSON.stringify(value ?? null)}
        onChange={(text) => onSet(JSON.parse(text))}
        options={options.map((option) =>
          [
            JSON.stringify(option),
            spec.optionLabels?.[(spec.options ?? []).indexOf(option)] ??
              (option === null ? "(none)" : option === "" ? "(empty)" : String(option)),
          ] as const
        )}
      />
    );
  }
  const commit = (text: string) => {
    const parsed = parseText(spec, text);
    if ("error" in parsed) {
      setBad(`needs ${parsed.error}`);
      return;
    }
    setBad(null);
    onSet(parsed.value);
  };
  return (
    <>
      <span className={spec.secret ? "secret" : undefined}>
        <TextInput
          size="sm"
          mono={spec.kind !== "string" || spec.secret}
          aria-label={spec.name}
          type={spec.secret && !revealed ? "password" : "text"}
          autoComplete={spec.secret ? "off" : undefined}
          placeholder={spec.placeholder ??
            (spec.nullable ? "(none)" : spec.kind === "path" ? "path" : "")}
          value={asText(value)}
          invalid={!!bad}
          onCommit={commit}
          onFocus={(event) => spec.secret && event.target.value === HIDDEN && event.target.select()}
        />
        {spec.secret && (
          <button
            type="button"
            className="reveal"
            title="Saved; never shown again. Type a new one to replace it."
            onClick={() => setRevealed(!revealed)}
          >
            {revealed ? "hide" : "show"}
          </button>
        )}
      </span>
      {bad && <span className="cfg-bad">{bad}</span>}
    </>
  );
}

export function ConfigModal(
  { name, onSaved, onClose }: {
    name: string;
    onSaved: (saved: Saved) => void;
    onClose: () => void;
  },
) {
  const [global, setGlobal] = useState<GlobalConfig | null>(null);
  const [modules, setModules] = useState<BlueprintConfig | null>(null);
  const [recommended, setRecommended] = useState<Recommended[]>([]);
  const [query, setQuery] = useState("");
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getJson<GlobalConfig>("global-config"),
      getJson<BlueprintConfig>(`blueprints/${encodeURIComponent(name)}/config`),
      recommendedOf(name),
    ]).then(([g, m, r]) => {
      setGlobal(g);
      setModules(m);
      setRecommended(r);
    }, (caught) => setProblem(errorText(caught)));
  }, [name]);

  const savedGlobal = global?.overrides ?? {};
  const savedModules = modules?.overrides ?? {};
  useEffect(() => {
    if (global && modules) {
      onSaved({ global: savedGlobal, modules: savedModules, defaults: global.defaults ?? {} });
    }
    // report what's saved whenever it changes
  }, [global, modules]);

  const save = async (action: () => Promise<void>) => {
    setProblem(null);
    try {
      await action();
    } catch (caught) {
      setProblem(errorText(caught));
    }
  };
  const saveModules = async (overrides: ModuleOverrides) =>
    setModules(
      await send<BlueprintConfig>("PUT", `blueprints/${encodeURIComponent(name)}/config`, {
        overrides,
      }),
    );
  const saveGlobal = async (overrides: Overrides) =>
    setGlobal(await send<GlobalConfig>("PUT", "global-config", { overrides }));
  const setField = (spec: FieldSpec, value: unknown) =>
    save(() => {
      if (spec.scope === "module") {
        return saveModules(
          withModule(savedModules, spec, same(value, spec.base) ? undefined : value),
        );
      }
      const next = { ...savedGlobal };
      if (same(value, spec.base)) {
        delete next[spec.id];
      } else {
        next[spec.id] = value;
      }
      return saveGlobal(next);
    });
  const resetField = (spec: FieldSpec) =>
    save(() => {
      if (spec.scope === "module") {
        return saveModules(withModule(savedModules, spec, undefined));
      }
      const { [spec.id]: _removed, ...rest } = savedGlobal;
      return saveGlobal(rest);
    });
  const valueOf = (spec: FieldSpec) =>
    spec.scope === "module"
      ? savedModules[spec.module!]?.[spec.name] ?? spec.base
      : spec.id in savedGlobal
      ? savedGlobal[spec.id]
      : spec.base;
  const isSaved = (spec: FieldSpec) =>
    spec.scope === "module"
      ? spec.name in (savedModules[spec.module!] ?? {})
      : spec.id in savedGlobal;

  const gFields = global ? globalFields(global) : [];
  const mFields = modules ? moduleFields(modules) : [];

  const keyValue = (key: string) => {
    if (key.includes(".")) {
      const at = key.lastIndexOf(".");
      const [module, field] = [key.slice(0, at), key.slice(at + 1)];
      const saved = savedModules[module];
      if (saved && field in saved) {
        return saved[field];
      }
      return mFields.find((spec) => spec.module === module && spec.name === field)?.base;
    }
    return key in savedGlobal ? savedGlobal[key] : global?.defaults?.[key];
  };
  const setMany = (values: Record<string, unknown>) =>
    save(async () => {
      const globals = { ...savedGlobal };
      let moduleNext = savedModules;
      for (const [key, value] of Object.entries(values)) {
        if (key.includes(".")) {
          const spec = mFields.find((one) => `${one.module}.${one.name}` === key);
          if (spec) {
            moduleNext = withModule(moduleNext, spec, same(value, spec.base) ? undefined : value);
          }
        } else if (same(value, global?.defaults?.[key])) {
          delete globals[key];
        } else {
          globals[key] = value;
        }
      }
      if (!same(moduleNext, savedModules)) {
        await saveModules(moduleNext);
      }
      if (!same(globals, savedGlobal)) {
        await saveGlobal(globals);
      }
    });
  const shown = (setting: Recommended) =>
    !setting.when ||
    Object.entries(setting.when).every(([key, value]) => same(keyValue(key), value));

  const row = (spec: FieldSpec, id: string, control: ReactNode, compact = true) => {
    const set = isSaved(spec);
    return (
      <div
        key={id}
        className={`cfg-row${set ? " set" : ""}`}
        data-field={id}
        title={spec.description ?? undefined}
      >
        <span className="k">
          <span className="n">{spec.label ?? spec.name.replaceAll("_", " ")}</span>
          {set && <span className="mark">saved</span>}
        </span>
        <span className="v">
          {control}
          {set
            ? <Reset spec={spec} onReset={() => resetField(spec)} />
            : <span className="reset-gap" />}
        </span>
        {spec.docs && (
          <a className="docs" href={spec.docs} target="_blank" rel="noopener noreferrer">
            How do I find this?
          </a>
        )}
        {!compact && spec.description && <span className="desc">{spec.description}</span>}
      </div>
    );
  };
  const field = (spec: FieldSpec) =>
    row(
      spec,
      spec.id,
      <Input spec={spec} value={valueOf(spec)} onSet={(value) => setField(spec, value)} />,
    );

  const recommendedRow = (setting: Recommended) => {
    if (setting.kind === "pick") {
      const choices = setting.choices ?? [];
      const current = choices.findIndex((choice) =>
        Object.entries(choice.set ?? {}).every(([key, value]) => same(keyValue(key), value))
      );
      return (
        <div key={setting.id} className="cfg-row pick" data-field={setting.id}>
          <span className="k">
            <span className="n">{setting.label}</span>
          </span>
          <span className="v">
            <Choose
              labels={choices.map((c) => c.label)}
              current={current}
              pick={(index) => setMany(choices[index].set ?? {})}
              name={setting.label}
            />
          </span>
        </div>
      );
    }
    const fields = [...mFields, ...gFields];
    const spec = fields.find((one) => one.id === setting.key && one.scope === setting.scope) ??
      fields.find((one) =>
        setting.scope === "module" && `${one.module}.${one.name}` === setting.key
      );
    if (!spec) {
      return null;
    }
    const labelled: FieldSpec = {
      ...spec,
      label: setting.label,
      placeholder: setting.placeholder ?? spec.placeholder,
      docs: setting.docs,
    };
    if (!setting.choices) {
      return row(
        labelled,
        setting.id,
        <Input spec={labelled} value={valueOf(spec)} onSet={(v) => setField(spec, v)} />,
      );
    }
    const choices = setting.choices;
    const current = choices.findIndex((choice) => same(choice.value, valueOf(spec)));
    return row(
      labelled,
      setting.id,
      <Choose
        labels={choices.map((c) => c.label)}
        current={current}
        pick={(index) => setField(spec, choices[index].value)}
        name={setting.label}
      />,
    );
  };

  const list = (fields: FieldSpec[], heading: (group: string) => string) => {
    const needle = query.trim().toLowerCase().replaceAll(" ", "_");
    const matching = fields.filter((spec) =>
      !needle || spec.id.toLowerCase().includes(needle) ||
      (spec.description ?? "").toLowerCase().includes(query.trim().toLowerCase())
    );
    if (matching.length === 0) {
      return (
        <div className="hint">{query ? `Nothing matches "${query}".` : "Nothing to set."}</div>
      );
    }
    const groups: { id: string; fields: FieldSpec[] }[] = [];
    for (const spec of matching) {
      if (groups.at(-1)?.id === spec.group) {
        groups.at(-1)!.fields.push(spec);
      } else {
        groups.push({ id: spec.group, fields: [spec] });
      }
    }
    return groups.map((group) => (
      <div
        key={group.id}
        className={`cfg-group${group.fields[0]?.scope === "module" ? " cfg-module" : ""}`}
        data-group={group.id}
      >
        <div className="cfg-gh">
          {group.fields[0]?.groupLabel ?? heading(group.id)}
          <span className="n">{String(group.fields.filter(isSaved).length || "")}</span>
        </div>
        {group.fields.map(field)}
      </div>
    ));
  };

  const recommendedRows = global && modules
    ? recommended.filter(shown).map(recommendedRow).filter(Boolean)
    : [];
  const count = Object.keys(savedGlobal).length +
    Object.values(savedModules).reduce((sum, fields) => sum + Object.keys(fields).length, 0);
  return (
    <Modal
      open
      onClose={onClose}
      size="lg"
      className="cfg"
      aria-label={`Configure ${name}`}
      title={
        <>
          Config <span className="cfg-sub">{count ? `${count} set` : "dimOS defaults"}</span>
        </>
      }
      footer={<Button onClick={onClose}>Done</Button>}
      data-bp-config
    >
      <TextInput
        className="cfg-search"
        type="search"
        icon="search"
        placeholder="Search config…"
        aria-label="search config"
        value={query}
        onChange={setQuery}
      />
      <p className="hint">Saved for every launch of this blueprint. Relaunch to apply.</p>
      {problem && <Alert tone="warn" className="cfg-problem">{problem}</Alert>}
      <div className="cfg-body">
        {recommendedRows.length > 0 && (
          <section className="cfg-sec recommended">
            <div className="cfg-sh">Recommended settings</div>
            <p className="hint">Decide these first: the rest can stay at dimOS's defaults.</p>
            {recommendedRows}
          </section>
        )}
        <section className="cfg-sec">
          <div className="cfg-sh">
            Module config<span className="of">{name}</span>
          </div>
          {modules
            ? list(mFields, (group) => group)
            : <div className="hint">reading {name}'s modules…</div>}
        </section>
        <section className="cfg-sec">
          <div className="cfg-sh">
            Global config<span className="of">every blueprint</span>
          </div>
          {global
            ? list(gFields, (group) => group === "general" ? "general" : `${group}_*`)
            : <div className="hint">reading dimOS's global config…</div>}
        </section>
      </div>
    </Modal>
  );
}
