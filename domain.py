"""Plot: Domain."""

import os
from pathlib import Path

import f90nml
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from matplotlib.patches import ConnectionPatch
from pyproj.proj import Proj
from salem import Map, gis, read_shapefile, wgs84
from shapely.geometry import LinearRing

# Path of namelist.wps.
filename = "data/namelists/namelist.wps.D03"
# Outer domain of right Axes.
domain = 2


def plot(
    ax: plt.Axes,
    args: f90nml.Namelist,
    pwrf: Proj,
    center: tuple[float],
    start: int,
) -> tuple[plt.Axes, gis.Grid]:
    """Plot on Single Axes."""
    filename = "/public/home/XiaAnRen/share/data/ChinaShp202503/Province.shp"
    china_map = read_shapefile(filename)

    e, n = center

    parent_id = args["parent_id"]
    parent_ratio = args["parent_grid_ratio"]
    i_start = args["i_parent_start"]
    j_start = args["j_parent_start"]
    e_we = args["e_we"]
    e_sn = args["e_sn"]
    dx = args["dx"]
    dy = args["dy"]

    # LL corner
    nx, ny = e_we[0] - 1, e_sn[0] - 1
    x0 = -(nx - 1) / 2.0 * dx + e  # -2 because of staggered grid
    y0 = -(ny - 1) / 2.0 * dy + n

    # parent grid
    grid = gis.Grid(nxny=(nx, ny), x0y0=(x0, y0), dxdy=(dx, dy), proj=pwrf)

    # child grids
    out = [grid]
    dxs = [dx]
    zips = zip(i_start, j_start, parent_id, parent_ratio, e_we, e_sn)
    for ips, jps, pid, ratio, we, sn in zips:
        if ips == 1:
            continue
        nx = (we - 1) / ratio
        ny = (sn - 1) / ratio

        prevgrid = out[pid - 1]
        xx, yy = prevgrid.corner_grid.x_coord, prevgrid.corner_grid.y_coord
        dx = prevgrid.dx / ratio
        dy = prevgrid.dy / ratio
        grid = gis.Grid(
            nxny=((we - 1), (sn - 1)),
            x0y0=(xx[ips - 1], yy[jps - 1]),
            dxdy=(dx, dy),
            pixel_ref="corner",
            proj=pwrf,
        )
        out.append(grid.center_grid)
        dxs.append(dx)

    maps = []
    for i, g in enumerate(out):
        m = Map(g)
        for j in range(i + 1, len(out)):
            cg = out[j]
            left, right, bottom, top = cg.extent
            s = [(left, bottom), (right, bottom), (right, top), (left, top)]
            m.set_geometry(LinearRing(s), crs=cg.proj, linewidth=1.625)
        maps.append(m)

    if start == 1:
        maps[0].set_lonlat_contours(xinterval=5, yinterval=5)
    maps[0].set_shapefile(countries=False)
    maps[0].set_shapefile(shape=china_map, linewidth=0.5)
    maps[0].set_rgb(natural_earth="hr")
    maps[0].plot(ax)

    left0, right0, bottom0, top0 = out[0].extent
    left, right, bottom, top = out[1].extent

    ax.text(
        (left0 - left0) / (right0 - left0) + 0.0075,
        (top0 - bottom0) / (top0 - bottom0) - 0.0075,
        f"$\\mathrm{{ D{start:02} ( {dxs[0] / 1000:g} km ) }}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        color="white",
        alpha=0.85,
    )
    ax.text(
        (left - left0) / (right0 - left0),
        (top - bottom0) / (top0 - bottom0) + 0.0075,
        f"$\\mathrm{{ D{start + 1:02} ( {dxs[1] / 1000:g} km ) }}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        color="white",
        alpha=0.85,
    )

    ax.tick_params(labelsize=16)

    return ax, out


def add_sites(ax: plt.Axes, maps: Map) -> plt.Axes:
    """Add Distribution of SURF & WPRD & Points & Coverage."""
    # SURF.
    filepath = Path("/data3/gengruomei/radar")
    folder = "surf"
    filename = "SURF_CHN_MUL_HOR_20250303000000.txt"

    file = filepath / folder / filename
    names = ["lats", "lons"]
    data = pd.read_csv(file, sep=" ", names=names, usecols=[3, 4], skiprows=2)

    lons, lats = data["lons"], data["lats"]
    lons, lats = maps.grid.transform(lons, lats)
    ax.scatter(lons, lats, c="black", s=0.5, alpha=0.5)

    # WPRD.
    filepath = Path("/data3/gengruomei/radar/wprd/WPRD个例数据/20250303")
    lons, lats = [], []
    for folder in filepath.iterdir():
        file = next(folder.iterdir())
        data = pd.read_csv(file, sep=" ", header=None, skiprows=1, nrows=1)
        lons.append(data.iloc[0, 1]), lats.append(data.iloc[0, 2])
    lons, lats = maps.grid.transform(lons, lats)
    ax.scatter(lons, lats, c="red", s=5, alpha=0.5)

    # Radar Site Points.
    points = [
        {"lon": 118.11581597619593, "lat": 35.64568551155871},  # AnDi
        {"lon": 117.55550469214923, "lat": 35.43144884412909},  # TangCun
        {"lon": 118.86289768825819, "lat": 35.78850995651179},  # QingFengLing
        {"lon": 118.08834973678188, "lat": 36.14557106889448},  # TianZhuang
        {"lon": 118.29159990844587, "lat": 35.96978713664454},  # DongLiDian
        {"lon": 117.71480888075074, "lat": 35.94232089723049},  # JingDou
    ]
    slon, slat = ([point[key] for point in points] for key in ("lon", "lat"))
    lons, lats = maps.grid.transform(slon, slat)
    ax.scatter(lons, lats, s=2, c="white", alpha=0.625)

    # Radar Scanning Coverage.
    coverage = GeoDataFrame(
        points,
        geometry=points_from_xy(slon, slat),
        crs="EPSG:4326",
    ).to_crs("EPSG:32649")
    coverage = GeoSeries(
        coverage.geometry.buffer(45000).unary_union,
        crs="EPSG:32649",
    ).to_crs("EPSG:4326")
    geom = next(iter(coverage.geometry))
    lons, lats = maps.grid.transform(geom.exterior.xy[0], geom.exterior.xy[1])
    ax.plot(lons, lats, c="white", alpha=0.625)
    return ax


def main() -> None:
    """Plot."""
    os.chdir(Path(__file__).resolve().parent)

    file = f90nml.read(filename)
    args = file["geogrid"]

    # define projection
    if args["map_proj"] == "lambert":
        pwrf = (
            "+proj=lcc +lat_1={truelat1} +lat_2={truelat2} "
            "+lat_0={ref_lat} +lon_0={stand_lon} "
            "+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    elif args["map_proj"] == "mercator":
        pwrf = (
            "+proj=merc +lat_ts={truelat1} +lon_0={stand_lon} "
            "+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    elif args["map_proj"] == "polar":
        pwrf = (
            "+proj=stere +lat_ts={truelat1} +lat_0=90.0 +lon_0={stand_lon} "
            "+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    else:
        error = f"WRF proj not implemented yet: {args['map_proj']}"
        raise NotImplementedError(error)
    pwrf = gis.check_crs(pwrf.format(**args))

    # get easting and northings from dom center (probably unnecessary here)
    e, n = gis.transform_proj(wgs84, pwrf, args["ref_lon"], args["ref_lat"])

    # Create Figure & Axes.
    fig = plt.figure(figsize=(12, 5))
    subfigs = fig.subfigures(1, 2, wspace=-0.25)
    for subfig in subfigs:
        subfig.set_facecolor("none")
    axes = [subfig.subplots(1, 1) for subfig in subfigs]

    # Axes: D01.
    _, out = plot(axes[0], args, pwrf, (e, n), start=1)

    # Process for D02.
    for _ in range(domain - 1):
        args["dx"] /= args["parent_grid_ratio"][1]
        args["dy"] /= args["parent_grid_ratio"][1]
        for key in ("e_we", "e_sn"):
            args[key].pop(0)
        for key in ("parent_grid_ratio", "i_parent_start", "j_parent_start"):
            args[key].pop(1)
        args["parent_id"].pop(-1)

    left0, right0, bottom0, top0 = out[0].extent
    left, right, bottom, top = out[domain - 1].extent

    # Axes: D02.
    e, n = (left + right) / 2, (top + bottom) / 2
    ax, out = plot(axes[1], args, pwrf, (e, n), start=domain)
    add_sites(ax, Map(next(iter(out))))

    # Connecting Lines.
    con = ConnectionPatch(
        xyA=(
            (right - left0) / (right0 - left0),
            (top - bottom0) / (top0 - bottom0),
        ),
        coordsA=axes[0].transAxes,
        xyB=(0, 1),
        coordsB=axes[1].transAxes,
    )
    subfig.add_artist(con)

    con = ConnectionPatch(
        xyA=(
            (right - left0) / (right0 - left0),
            (bottom - bottom0) / (top0 - bottom0),
        ),
        coordsA=axes[0].transAxes,
        xyB=(0, 0),
        coordsB=axes[1].transAxes,
    )
    subfig.add_artist(con)

    fig.savefig("images/domain.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
