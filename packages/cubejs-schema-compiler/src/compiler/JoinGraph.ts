import R from 'ramda';
import Graph from 'node-dijkstra';

import type { CubeValidator } from './CubeValidator';
import type { CubeEvaluator, EvaluatedCubeMeasures, MeasureDefinition } from './CubeEvaluator';
import type { CubeDefinition, CubeJoinDefinition } from './CubeSymbols';
import type { ErrorReporter } from './ErrorReporter';
import { UserError } from './UserError';

type JoinEdge = {
  join: CubeJoinDefinition,
  from: string,
  to: string,
  originalFrom: string,
  originalTo: string,
};

// edge name (string like 'from-to') -> edge
type Edges = Record<string, JoinEdge>;

type JoinTreeJoins = Array<JoinEdge>;

type JoinTree = {
  root: string,
  joins: JoinTreeJoins,
};

export type FinishedJoinTree = JoinTree & {
  multiplicationFactor: Record<string, boolean>,
};

export type JoinHint = string | Array<string>;

export type JoinHints = Array<JoinHint>;

function present<T>(t: T | null): t is T {
  return t !== null;
}

export class JoinGraph {
  // source node -> destination node -> weight
  protected nodes: Record<string, Record<string, 1>>;

  protected edges: Edges;

  // source node -> destination node -> weight
  protected undirectedNodes: Record<string, Record<string, 1>>;

  protected builtJoins: Record<string, FinishedJoinTree>;

  protected graph: Graph | null;

  protected cachedConnectedComponents: Record<string, number> | null;

  public constructor(protected cubeValidator: CubeValidator, protected cubeEvaluator: CubeEvaluator) {
    this.nodes = {};
    this.edges = {};
    this.undirectedNodes = {};
    this.builtJoins = {};
    this.graph = null;
    this.cachedConnectedComponents = null;
  }

  public compile(cubes: unknown, errorReporter: ErrorReporter): void {
    this.edges = R.compose<
        [Array<CubeDefinition>],
        Array<CubeDefinition>,
        Array<Array<[string, JoinEdge]>>,
        Array<[string, JoinEdge]>,
        Record<string, JoinEdge>
    >(
      R.fromPairs,
      R.unnest,
      R.map(v => this.buildJoinEdges(v, errorReporter.inContext(`${v.name} cube`))),
      R.filter(this.cubeValidator.isCubeValid.bind(this.cubeValidator))
    )(this.cubeEvaluator.cubeList);
    // This requires @types/ramda@0.29 or newer
    // @ts-ignore
    this.nodes = R.compose<
        [Record<string, JoinEdge>],
        Array<[string, JoinEdge]>,
        Array<JoinEdge>,
        Record<string, Array<JoinEdge> | undefined>,
        Record<string, Record<string, 1>>
    >(
      // This requires @types/ramda@0.29 or newer
      // @ts-ignore
      R.map(groupedByFrom => R.fromPairs((groupedByFrom ?? []).map(join => [join.to, 1]))),
      R.groupBy(join => join.from),
      R.map(v => v[1]),
      R.toPairs
    )(this.edges);
    // This requires @types/ramda@0.29 or newer
    // @ts-ignore
    this.undirectedNodes = R.compose<
        [Record<string, JoinEdge>],
        Array<[string, JoinEdge]>,
        Array<[JoinEdge, {from: string, to: string }]>,
        Array<{from: string, to: string }>,
        Record<string, Array<{from: string, to: string }> | undefined>,
        Record<string, Record<string, 1>>
    >(
      // This requires @types/ramda@0.29 or newer
      // @ts-ignore
      R.map(groupedByFrom => R.fromPairs((groupedByFrom ?? []).map(join => [join.from, 1]))),
      R.groupBy(join => join.to),
      R.unnest,
      R.map(v => [v[1], { from: v[1].to, to: v[1].from }]),
      R.toPairs
    )(this.edges);
    this.graph = new Graph(this.nodes);
  }

  protected buildJoinEdges(cube: CubeDefinition, errorReporter: ErrorReporter): Array<[string, JoinEdge]> {
    return R.compose<
        [Record<string, CubeJoinDefinition>],
        Array<[string, CubeJoinDefinition]>,
        Array<[string, CubeJoinDefinition] | null>,
        Array<[string, CubeJoinDefinition]>,
        Array<[string, JoinEdge]>,
        Array<[string, JoinEdge] | null>,
        Array<[string, JoinEdge]>
    >(
      R.filter<
          [string, JoinEdge] | null,
          [string, JoinEdge]
      >(present),
      R.map(join => {
        const multipliedMeasures = R.compose<
            [EvaluatedCubeMeasures],
            Array<MeasureDefinition>,
            Array<MeasureDefinition>
        >(
          R.filter<MeasureDefinition>(
            m => m.sql && this.cubeEvaluator.funcArguments(m.sql).length === 0 && m.sql() === 'count(*)' ||
            ['sum', 'avg', 'count', 'number'].indexOf(m.type) !== -1
          ),
          R.values
        );
        const joinRequired =
          (v) => `primary key for '${v}' is required when join is defined in order to make aggregates work properly`;
        if (
          !this.cubeEvaluator.primaryKeys[join[1].from].length &&
          multipliedMeasures(this.cubeEvaluator.measuresForCube(join[1].from)).length > 0
        ) {
          errorReporter.error(joinRequired(join[1].from));
          return null;
        }
        if (!this.cubeEvaluator.primaryKeys[join[1].to].length &&
          multipliedMeasures(this.cubeEvaluator.measuresForCube(join[1].to)).length > 0) {
          errorReporter.error(joinRequired(join[1].to));
          return null;
        }
        return join;
      }),
      R.map(join => [`${cube.name}-${join[0]}`, {
        join: join[1],
        from: cube.name,
        to: join[0],
        originalFrom: cube.name,
        originalTo: join[0]
      }]),
      R.filter<
          [string, CubeJoinDefinition] | null,
          [string, CubeJoinDefinition]
      >(present),
      R.map(join => {
        if (!this.cubeEvaluator.cubeExists(join[0])) {
          errorReporter.error(`Cube ${join[0]} doesn't exist`);
          return null;
        }
        return join;
      }),
      R.toPairs
    )(cube.joins || {});
  }

  protected buildJoinNode(cube: CubeDefinition): Record<string, 1> {
    return R.compose<
      [Record<string, CubeJoinDefinition>],
      Array<[string, CubeJoinDefinition]>,
      Array<[string, 1]>,
      Record<string, 1>
    >(
      R.fromPairs,
      R.map(v => [v[0], 1]),
      R.toPairs
    )(cube.joins || {});
  }

  public buildJoin(cubesToJoin: JoinHints): FinishedJoinTree | null {
    if (!cubesToJoin.length) {
      return null;
    }
    const key = JSON.stringify(cubesToJoin);
    if (!this.builtJoins[key]) {
      const join = R.pipe<
          [JoinHints],
          Array<JoinTree | null>,
          Array<JoinTree>,
          Array<JoinTree>
      >(
        R.map(
          cube => this.buildJoinTreeForRoot(cube, R.without([cube], cubesToJoin))
        ),
        R.filter<
          JoinTree | null,
          JoinTree
        >(present),
        R.sortBy(joinTree => joinTree.joins.length)
      )(cubesToJoin)[0];
      if (!join) {
        throw new UserError(`Can't find join path to join ${cubesToJoin.map(v => `'${v}'`).join(', ')}`);
      }
      this.builtJoins[key] = {
        ...join,
        multiplicationFactor: R.compose<
          [JoinHints],
          Array<[string, boolean]>,
          Record<string, boolean>
        >(
          R.fromPairs,
          R.map(v => [this.cubeFromPath(v), this.findMultiplicationFactorFor(this.cubeFromPath(v), join.joins)])
        )(cubesToJoin)
      };
    }
    return this.builtJoins[key];
  }

  protected cubeFromPath(cubePath: string | Array<string>): string {
    if (Array.isArray(cubePath)) {
      return cubePath[cubePath.length - 1];
    }
    return cubePath;
  }

  protected buildJoinTreeForRoot(root: JoinHint, cubesToJoin: JoinHints): JoinTree | null {
    const self = this;
    const { graph } = this;
    if (graph === null) {
      // JoinGraph was not compiled
      return null;
    }
    if (Array.isArray(root)) {
      const [newRoot, ...additionalToJoin] = root;
      cubesToJoin = [additionalToJoin, ...cubesToJoin];
      root = newRoot;
    }

    const singleRoot = root;
    const nodesJoined = {};
    const result = cubesToJoin.map(joinHints => {
      if (!Array.isArray(joinHints)) {
        joinHints = [joinHints];
      }
      let prevNode = singleRoot;
      return joinHints.filter(toJoin => toJoin !== prevNode).map(toJoin => {
        if (nodesJoined[toJoin]) {
          prevNode = toJoin;
          return { joins: [] };
        }
        const path = graph.path(prevNode, toJoin);
        if (!path) {
          return null;
        }
        if (!Array.isArray(path)) {
          // Unexpected object return from graph, it should do so only when path cost was requested
          return null;
        }
        const foundJoins = self.joinsByPath(path);
        prevNode = toJoin;
        nodesJoined[toJoin] = true;
        return { cubes: path, joins: foundJoins };
      });
    }).flat().reduce<{joins: Array<[number, JoinEdge]>} | null>((joined, res) => {
      if (res === null || joined === null) {
        return null;
      }
      const indexedPairs = R.compose<
        [Array<JoinEdge>],
        Array<[number, JoinEdge]>
      >(
        R.addIndex(R.map)((j, i) => [i + joined.joins.length, j])
      );
      return {
        joins: [...joined.joins, ...indexedPairs(res.joins)],
      };
    }, { joins: [] });

    if (!result) {
      return null;
    }

    const pairsSortedByIndex =
      R.compose<
        [Array<[number, JoinEdge]>],
        Array<[number, JoinEdge]>,
        Array<JoinEdge>,
        Array<JoinEdge>
      >(R.uniq, R.map(indexToJoin => indexToJoin[1]), R.sortBy(indexToJoin => indexToJoin[0]));
    return {
      joins: pairsSortedByIndex(result.joins),
      root: singleRoot
    };
  }

  protected findMultiplicationFactorFor(cube: string, joins: JoinTreeJoins): boolean {
    const visited = {};
    const self = this;
    function findIfMultipliedRecursive(currentCube: string): boolean {
      if (visited[currentCube]) {
        return false;
      }
      visited[currentCube] = true;
      function nextNode(nextJoin: JoinEdge): string {
        return nextJoin.from === currentCube ? nextJoin.to : nextJoin.from;
      }
      const nextJoins = joins.filter(j => j.from === currentCube || j.to === currentCube);
      if (nextJoins.find(
        nextJoin => self.checkIfCubeMultiplied(currentCube, nextJoin) && !visited[nextNode(nextJoin)]
      )) {
        return true;
      }
      return !!nextJoins.find(
        nextJoin => findIfMultipliedRecursive(nextNode(nextJoin))
      );
    }
    return findIfMultipliedRecursive(cube);
  }

  protected checkIfCubeMultiplied(cube: string, join: JoinEdge): boolean {
    return join.from === cube && join.join.relationship === 'hasMany' ||
      join.to === cube && join.join.relationship === 'belongsTo';
  }

  protected joinsByPath(path: Array<string>): Array<JoinEdge> {
    return R.range(0, path.length - 1).map(i => this.edges[`${path[i]}-${path[i + 1]}`]);
  }

  public connectedComponents(): Record<string, number> {
    if (!this.cachedConnectedComponents) {
      let componentId = 1;
      const components = {};
      R.toPairs(this.nodes).map(nameToConnection => nameToConnection[0]).forEach(node => {
        this.findConnectedComponent(componentId, node, components);
        componentId += 1;
      });
      this.cachedConnectedComponents = components;
    }
    return this.cachedConnectedComponents;
  }

  protected findConnectedComponent(componentId: number, node: string, components: Record<string, number>): void {
    if (!components[node]) {
      components[node] = componentId;
      R.toPairs(this.undirectedNodes[node])
        .map(connectedNodeNames => connectedNodeNames[0])
        .forEach(connectedNode => {
          this.findConnectedComponent(componentId, connectedNode, components);
        });
    }
  }
}
