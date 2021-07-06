import argparse
import gc
import io
import json
import os
import random
import sqlite3
from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import product

import matplotlib.pyplot as plt
from diffractio import degrees, np, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_sources_XY import Scalar_source_XY
from skimage import color
from tqdm import tqdm


def create_db(db_dir):
    """
    Create a sqlite database file.
    
    :param db_dir: directory for the database including the file name
    :return: sqlite connection to the database where the CRPs are to be stored
    """
    try:
        conn = sqlite3.connect(db_dir)
        sql_create_sim_table = '''CREATE TABLE simulation (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    challenge TEXT,
                                    response TEXT
                                  );'''
        cursor = conn.cursor()
        cursor.execute(sql_create_sim_table)
        return conn
    except sqlite3.Error as e:
        print(e)
        exit()


def insert_crp_in_db(conn, challenge, response):
    """
    Stores a CRP in the database.

    :param conn: sqlite connection to the database where the CRPs are to be stored
    :param challenge: challenge to be stored
    :param response: response to be stored
    """
    sql_insert_data = "INSERT INTO simulation(challenge,response) VALUES(?,?)"
    cursor = conn.cursor()
    cursor.execute(sql_insert_data, (challenge, response))


def compute_crps(conn, puf, x0, y0, container_size, measure_dist, wavelength, blocks_per_row, max_crps, random_measure,
                 only_every_2nd):
    """
    Computes and stores the CRPs for the given simulated PUF.

    :param conn: sqlite connection to the database where the CRPs are to be stored
    :param puf: puf simulation object
    :param x0: x-axis of the simulated PUF
    :param y0: y-axis of the simulated PUF
    :param container_size: height/width of the simulated PUF
    :param measure_dist: distance behind the PUF borders where the electric field is evaluated
    :param wavelength: wavelength to be used for the laser
    :param blocks_per_row: number of blocks/bits per row
    :param max_crps: number of crps that are generated
    :param random_measure: whether to randomly choose the challenges
    :param only_every_2nd: whether to use only every other block in a row
    """
    c_size = blocks_per_row ** 2
    lcd_array_size = container_size / 4
    lcd_start = lcd_array_size // 2
    block_length = lcd_array_size // blocks_per_row
    challenge_space_size = 2 ** (blocks_per_row * blocks_per_row)

    if max_crps == 0:
        max_crps = challenge_space_size

    if random_measure:
        if only_every_2nd:
            challenge_space = [np.random.randint(2, size=(int(np.ceil(blocks_per_row ** 2 / 2)))) for _ in
                               range(max_crps)]
        else:
            challenge_space = [np.random.randint(2, size=blocks_per_row ** 2) for _ in range(max_crps)]
    else:
        if only_every_2nd:
            challenge_space = product(range(2), repeat=(int(np.ceil(blocks_per_row ** 2 / 2))))
        else:
            challenge_space = product(range(2), repeat=blocks_per_row ** 2)

    for challenge in tqdm(challenge_space, total=max_crps):
        challenge = np.array([int(bit) for bit in challenge])
        if only_every_2nd:
            challenge = np.insert(challenge, range(1, len(challenge), 1), 0)

        if c_size != 0:
            number_set_bits = sum(bit for bit in challenge)
            for idx in range(number_set_bits - c_size):
                set_bits_idxs = [idx for idx, bit in enumerate(challenge) if bit]
                challenge[random.choice(set_bits_idxs)] = 0

        challenge_id = "".join(str(bit) for bit in challenge)

        challenge_mask = Scalar_mask_XY(x0, y0, wavelength)

        for bit, value in enumerate(challenge):
            if value:
                bit_mask = Scalar_mask_XY(x0, y0, wavelength)
                bit_x_pos = bit % blocks_per_row
                bit_y_pos = bit // blocks_per_row
                block_x_pos = -lcd_start + bit_x_pos * block_length + block_length / 2
                block_y_pos = -lcd_start + bit_y_pos * block_length + block_length / 2
                block_pos = (block_x_pos * um, block_y_pos * um)
                block_size = (block_length * um, block_length * um)
                bit_mask.square(r0=block_pos, size=block_size, angle=0 * degrees)
                challenge_mask = challenge_mask + bit_mask

        beam = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        beam.gauss_beam(A=100,
                        r0=(0, 0),
                        w0=(container_size * um, container_size * um),
                        theta=0 * degrees,
                        phi=-0 * degrees)

        puf.incident_field(beam * challenge_mask)
        puf.BPM()

        puf.draw_XY(z0=container_size + measure_dist, logarithm=True)
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig = plt.gca().figure

        io_buf = io.BytesIO()
        fig.savefig(io_buf, bbox_inches='tight', pad_inches=0, format='raw')
        io_buf.seek(0)

        response = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        response = np.reshape(response, newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        response = color.rgb2gray(color.rgba2rgb(response))
        response = json.dumps(response.tolist())

        io_buf.close()
        insert_crp_in_db(conn, challenge_id, response)
        plt.close("all")

        puf.clear_field()
        gc.collect()


def run_simulation(root, blocks_per_row, max_crps, multiplier, puf, x0, y0, container_size, measure_dist, wavelength):
    """
    Runs a simulation of a PUF for the provided parameters and computes and stores the generated CRPs.

    :param root: root folder directory where datasets are to be stored
    :param blocks_per_row: number of blocks/bits per row
    :param max_crps: number of crps that are generated
    :param multiplier: multiplier for the x and y dimensions of the PUF
    :param puf: puf simulation object
    :param x0: x-axis of the simulated PUF
    :param y0: y-axis of the simulated PUF
    :param container_size: height/width of the simulated PUF
    :param measure_dist: distance behind the PUF borders where the electric field is evaluated
    :param wavelength: wavelength to be used for the laser
    """
    bits = blocks_per_row ** 2

    # type A, C and D
    for c_size in (bits, bits // 2, int(bits * 2 / 3)):

        db_name = f"{root}/{bits}bx{multiplier}_mr{max_crps}"
        if c_size < bits:
            db_name += f"_mb{c_size}"
        db_name += ".db"

        if os.path.isfile(db_name):
            print(f"{db_name} already exists!")
            continue

        conn = create_db(db_name)

        compute_crps(conn, puf, x0, y0, container_size, measure_dist, wavelength, blocks_per_row, max_crps,
                     random_measure=True, only_every_2nd=False)

        conn.commit()
        conn.close()

    # type B
    if blocks_per_row % 2 == 1:
        db_name = f"{root}/{bits}bx{multiplier}_mr{max_crps}_2nd.db"
        if os.path.isfile(db_name):
            print(f"{db_name} already exists!")
            return

        conn = create_db(db_name)

        compute_crps(conn, puf, x0, y0, container_size, measure_dist, wavelength, blocks_per_row, max_crps,
                     random_measure=True, only_every_2nd=True)
        conn.commit()
        conn.close()


def create_spheres(bounds, minr, maxr, padd, multiplier, sphere_file):
    """
    Runs a js script that creates the scatterers (spheres) which will be used for the PUF simulation.

    :param bounds: boundary size for the PUF
    :param minr: minimum radius of a scatterer
    :param maxr: maximum radius of a scatterer
    :param padd: padding of a scatterer
    :param multiplier: multiplier for the boundaries of the PUF
    :param sphere_file: file where to store the spheres
    """
    cmd = f"node create_spheres.js --multiplier={multiplier} --bounds={bounds} --minr={minr} --maxr={maxr} --padd={padd} --file={sphere_file}"
    os.system(cmd)


def add_spheres_to_PUF(puf, sphere_file):
    """
    Adds all scatterers (spheres) to the simulated PUF.

    :param puf: puf simulation object
    :param sphere_file: file where the scatterer details are stored
    :return: puf simulation object with scatterers
    """
    spheres = []
    with open(sphere_file) as sphere_file:
        col_names = ('x', 'y', 'z', 'radius')
        for line in sphere_file:
            sphere_data = [float(data.strip()) for data in line.split(",")]
            sphere = {}

            for i, col_name in enumerate(col_names):
                sphere[col_name] = sphere_data[i]

            spheres.append(sphere)
    print("Spheres opened")

    for sphere in tqdm(spheres):
        radius = sphere["radius"]
        puf.sphere(
            r0=(sphere["x"], sphere["y"], sphere["z"]),
            radius=(radius, radius, radius),
            refraction_index=1.52,
            angles=0)

    return puf


def main():
    """
    Runs the integrated optical PUF simulations and generates the datasets. By default, all datasets with an uneven
    number of bits in a row starting from 5 up to 15 are generated, but additional arguments enable finer adjustments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=1,
                        help="number of processes that will be run for the attacks.")
    parser.add_argument('--root', '--root_folder', required=True,
                        help="root folder directory where datasets are to be stored.")
    parser.add_argument('--multiply', type=int, default=1,
                        help="multiplier for the x and y dimensions of the PUF.")
    parser.add_argument('--max', type=int, default=0,
                        help="number of crps that are generated. default: exhaustively generates all crps!")
    parser.add_argument('--step', type=int, default=1,
                        help="step size between the number of bits in a row between all simulated PUFs.")
    parser.add_argument('--with-even', default=False,
                        help="whether to include even number of bits in a row.")

    args = parser.parse_args()
    multiplier = args.multiply
    max_crps = args.max
    processes = args.p
    with_even = args.with_even

    container_size = 512 * multiplier * um
    container_depth = 512 * um
    measure_dist = 300 * um
    x_y_resolution = 512 * multiplier
    z_resolution = 512
    border_value = container_size / 2
    wavelength = 1

    x0 = np.linspace(-border_value, border_value, x_y_resolution)
    y0 = np.linspace(-border_value, border_value, x_y_resolution)
    z0 = np.linspace(0, container_depth + measure_dist, z_resolution)

    puf = Scalar_mask_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)
    puf = add_spheres_to_PUF(puf, "Spheres.txt")

    start = 4 if with_even else 5
    end = 17
    step = args.step
    if not with_even:
        step *= 2

    if processes == 1:
        for blocks_per_row in range(start, end, step):
            run_simulation(args.root, blocks_per_row, max_crps, multiplier, puf, x0, y0, container_size, measure_dist,
                           wavelength)
    else:
        pool = Pool(max_workers=processes)
        with tqdm(total=end - start) as progress:
            futures = []
            for blocks_per_row in range(start, end, step):
                future = pool.submit(run_simulation, args.root, blocks_per_row, max_crps, multiplier, puf, x0, y0,
                                     container_size, measure_dist, wavelength)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            try:
                for f in futures:
                    f.result()
            except Exception as exc:
                print(exc)
                f.cancel()


if __name__ == "__main__":
    main()
