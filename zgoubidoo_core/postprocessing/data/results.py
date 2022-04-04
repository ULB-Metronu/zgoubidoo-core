import pandas


def results_to_df(results):
    values = []
    for idx, val in enumerate(results):
        values.append([
            val[0][0],
            val[0][1],
            val[0][2],
            val[1][0],
            val[1][1],
            val[1][2],
            val[2]]
        )

    df = pandas.DataFrame(values, columns=['X', 'Y', 'Z', 'UX', 'UY', 'UZ', 'Brho'])
    return df
