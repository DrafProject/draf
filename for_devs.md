
# For developers

## Common abbreviated programming constructs

| short | long |
|-------|------------------|
| `cs` | CaseStudy object |
| `sc`, `scen` | Scenario object |
| `m`, `mdl` | Model |
| `d`, `dims` | Dimension container object |
| `p`, `params` | Parameters container object |
| `v`, `vars` | Variables container object |
| `r`, `res` | Results container object |
| `ent` | Entity: a variable or parameter |
| `doc` | Documentation / description string |
| `constr` | Constraint |
| `meta` | Meta data |
| `df` | Pandas `DataFrame` |
| `ser` | Pandas `Series` |
| `fp` | file path |
| `gp` | `gurobipy` - the Gurobi Python Interface |

## Bump version

Bump version (replace `<part>` with `major`, `minor`, or `patch`) with:

```sh
bump2version --dry-run --verbose <part>
bump2version <part>
git push origin <tag_name>
```

Type annotations are used throughout the project.
