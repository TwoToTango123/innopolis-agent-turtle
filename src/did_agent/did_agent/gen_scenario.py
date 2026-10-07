"""Generate a scenario YAML. Works without ROS:

    python3 -m did_agent.gen_scenario hard --seed 7 -o scenarios/hard.yaml
    ros2 run did_agent gen_scenario hard --seed 7 -o /tmp/hard.yaml
"""
import argparse
import os

from did_agent.core.grid_map import GridMap
from did_agent.core.scenario import DIFFICULTY, generate

DEFAULT_MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'maps', 'map.yaml')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('difficulty', choices=sorted(DIFFICULTY))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--map', default=DEFAULT_MAP)
    ap.add_argument('-o', '--output', help='YAML path (default: print to stdout)')
    args = ap.parse_args(argv)
    if not os.path.exists(args.map):
        from ament_index_python.packages import get_package_share_directory
        args.map = os.path.join(get_package_share_directory('did_agent'), 'maps', 'map.yaml')
    sc = generate(args.difficulty, args.seed, GridMap.from_yaml(args.map))
    if args.output:
        sc.save(args.output)
        print(f'{args.output}: {len(sc.samples)} samples, {len(sc.terrain)} terrain zones, '
              f'{len(sc.hazards)} hazards, {len(sc.events)} events')
    else:
        import yaml
        print(yaml.safe_dump(sc.to_dict(), sort_keys=False))


if __name__ == '__main__':
    main()
